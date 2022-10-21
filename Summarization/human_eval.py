import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import pickle
import argparse
import gc
import time
import logging

gc.enable()

from datasets import load_metric
import spacy
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, PegasusConfig
from torch.cuda.amp import autocast as autocast
from torch.utils import data
from torch.utils.data import (
    SequentialSampler, RandomSampler
)
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.grad_scaler import ShardedGradScaler

from utils import *
from dataset import *
from dataset_finetune_entity import *
from dataset_finetune_summary import *
from engine_pretrain import *
from engine_finetune_entity import *
from engine_finetune_summary import *

from models_summarization.model_finetune import *
from models_summarization.model_soft import *
from models_summarization.model_mixture import *

from pathlib import Path


def set_args():
    parser = argparse.ArgumentParser(description="latentRE")

    root = "/data/mathieu/"
    data_root = "/data/mathieu/"

    # general stuff
    parser.add_argument("--seed", dest="seed", type=int,
                        default=42, help="seed for network")
    parser.add_argument("--cuda", dest="cuda", type=str,
                        default="2", help="gpu id")
    parser.add_argument("--local_rank", dest="local_rank", type=int,
                        default=-1, help="local rank")
    parser.add_argument("--exp_id", dest="exp_id", type=str,
                        default='001', help="id for current exp")
    parser.add_argument("--debug", action='store_true',
                        default=False, help="whether debug with breakpoint")
    parser.add_argument("--log_dir", dest="log_dir", type=str,
                        default='./log', help="The path to log dir")
    parser.add_argument("--log_name", dest="log_name", type=str,
                        default='dummy', help="The file name of log file")

    # data
    parser.add_argument("--data_dir", dest="data_dir", type=str,
                        default=data_root + "DATASETS/PromptSumm/")
    parser.add_argument("--dataset_name", dest="dataset_name", type=str,
                        default="xsum")
    parser.add_argument("--dataset_cache_dir", dest="dataset_cache_dir", type=str,
                        default="../../hf_datasets/", help="dataset cache folder")
    parser.add_argument("--few_shot", dest="few_shot", type=str,
                        default="10", help="number of data points for training AND validation")
    parser.add_argument("--zero_shot", action='store_true')
    parser.add_argument("--num_seeds", dest="num_seeds", type=int,
                        default=3, help="number of seeds to sample for training AND validation")
    parser.add_argument("--max_test_size", dest="max_test_size", type=int,
                        default=20000, help="max test set size")

    # model
    ##### base model
    parser.add_argument("--model", dest="model", type=str,
                        default="PegasusMixPrompt", choices=["T5Finetune", "T5SoftPrompt", "T5MixPrompt",
                        "BartFinetune", 'BartSoftPrompt', 'BartMixPrompt',
                        "PegasusFinetune", 'PegasusSoftPrompt', 'PegasusMixPrompt'])
    parser.add_argument("--model_name", dest="model_name", type=str,
                        default="google/pegasus-large",
                        choices=["google/t5-v1_1-large", "facebook/bart-large", "google/pegasus-large"])
    parser.add_argument("--use_lm_adapted", dest="use_lm_adapted", type=int,
                        default=0,
                        help="whether to use lm_adapted model")  # if we use bart, then automatically don't use lm_adapted
    parser.add_argument("--lm_adapted_path", dest="lm_adapted_path", type=str,
                        default=root + "lm_adapted_t5model/torch_ckpt/large/pytorch_model.bin",
                        help="The path of lm_adapted model")
    parser.add_argument("--cache_path", dest="cache_path", type=str,
                        default=root + "hf_models/pegasus-large/",
                        help="The path of huggingface cache: /data/mathieu/hf_models/t5-v1-large/, /data/ruochen/hf_models/bart-base/, /data/ruochen/hf_models/pegasus-large/")
    parser.add_argument("--tune_weights", dest="tune_weights", action='store_true',
                        default=False)
    # prompt
    parser.add_argument("--concat_mode", dest="concat_mode", type=str,
                        default="concat_right", choices=["concat_right", "concat_left"])
    parser.add_argument("--prompt_number", dest="prompt_number", type=int,
                        default=300, help="The number of prompt")
    ##### discrete prompt
    parser.add_argument("--guidance_type", dest="guidance_type", type=str,
                        default="ents")
    parser.add_argument("--separator", dest="separator", type=str,
                        default=",", choices=[",", " "])
    parser.add_argument("--guidance_mode", dest="guidance_mode", type=str,
                        default="input",
                        choices=["input", "input_most_frequent", "input_salient_sentences", "input_and_target",
                                 "target", "target_unique", 'target_unique_filtered'])
    parser.add_argument("--filter_type", dest="filter_type", type=str,
                        default=None,
                        choices=['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW',
                                 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'])
    parser.add_argument("--max_guidance_length", dest="max_guidance_length", type=int,
                        default=100)
    parser.add_argument("--counterfactual_removal", dest="counterfactual_removal", type=bool,
                        default=False,
                        help="whether to use counterfactual removal method during training to enforce causal link")

    # optimization
    parser.add_argument("--adam_epsilon", dest="adam_epsilon", type=float,
                        default=1e-8, help="adam epsilon")
    ##### pretraining
    parser.add_argument("--optimizer_pretrain", dest="optimizer_pretrain", type=str,
                        default="adafactor", help='optimizer for the pre-training')
    parser.add_argument("--lr_pretrain", dest="lr_pretrain", type=float,
                        default=5e-1, help='learning rate')
    parser.add_argument("--lr_pretrain_full_weights", dest="lr_pretrain_full_weights", type=float,
                        default=5e-5, help='learning rate if the pre-training changes all weights')
    parser.add_argument("--batch_size_per_gpu_pretrain", dest="batch_size_per_gpu_pretrain", type=int,
                        default=1, help="batch size per gpu")
    parser.add_argument("--valid_size_per_gpu_pretrain", dest="valid_size_per_gpu_pretrain", type=int,
                        default=4, help="valid size per gpu")
    parser.add_argument("--test_size_per_gpu_pretrain", dest="test_size_per_gpu_pretrain", type=int,
                        default=4, help="test size per gpu")
    parser.add_argument("--gradient_accumulation_steps_pretrain", dest="gradient_accumulation_steps_pretrain", type=int,
                        default=4, help="gradient accumulation steps")
    parser.add_argument("--max_epoch_pretrain", dest="max_epoch_pretrain", type=int,
                        default=5, help="max epoch number")
    parser.add_argument("--num_workers_pretrain", dest="num_workers_pretrain", type=int,
                        default=4, help="dataloader num_workers")
    parser.add_argument("--weight_decay_pretrain", dest="weight_decay_pretrain", type=float,
                        default=0, help="weight decay")
    parser.add_argument("--warmup_steps_pretrain", dest="warmup_steps_pretrain", type=float,
                        default=0.01, help="warmup steps")
    parser.add_argument("--max_grad_norm_pretrain", dest="max_grad_norm_pretrain", type=float,
                        default=1.0, help="max grad norm")
    parser.add_argument("--pretrain_dataset_path", dest="pretrain_dataset_path", type=str,
                        default="", help="pretrain data path when using huggingface dataset")
    parser.add_argument("--use_huggingface_dataset", dest="use_huggingface_dataset", action='store_true',
                        default=False, help="whether to use huggingface dataset for pretraining")
    parser.add_argument("--pretrain_with_ent_chain", dest="pretrain_with_ent_chain", action='store_true',
                        default=False, help="whether to pretrain with ent chain as input")
    parser.add_argument("--eval_step_pretrain", dest="eval_step_pretrain", type=int,
                        default=15000, help="how many steps to eval")
    ##### entity prompt tuning
    parser.add_argument("--lr_entity", dest="lr_entity", type=float,
                        default=5e-3, help='learning rate')
    parser.add_argument("--batch_size_per_gpu_entity", dest="batch_size_per_gpu_entity", type=int,
                        default=2, help="batch size per gpu")
    parser.add_argument("--valid_size_per_gpu_entity", dest="valid_size_per_gpu_entity", type=int,
                        default=4, help="valid size per gpu")
    parser.add_argument("--test_size_per_gpu_entity", dest="test_size_per_gpu_entity", type=int,
                        default=4, help="test size per gpu")
    parser.add_argument("--gradient_accumulation_steps_entity", dest="gradient_accumulation_steps_entity", type=int,
                        default=2, help="gradient accumulation steps")
    parser.add_argument("--max_epoch_entity", dest="max_epoch_entity", type=int,
                        default=60, help="max epoch number")
    parser.add_argument("--num_workers_entity", dest="num_workers_entity", type=int,
                        default=4, help="dataloader num_workers")
    parser.add_argument("--weight_decay_entity", dest="weight_decay_entity", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--warmup_steps_entity", dest="warmup_steps_entity", type=float,
                        default=0.01, help="warmup steps")
    parser.add_argument("--max_grad_norm_entity", dest="max_grad_norm_entity", type=float,
                        default=1.0, help="max grad norm")
    parser.add_argument("--eval_step_entity", dest="eval_step_entity", type=int,
                        default=15000, help="how many steps to eval")
    ##### summary prompt tuning
    parser.add_argument("--train_sample_summary", dest="train_sample_summary", type=bool,
                        default=True, help="dynamic sample or not")
    parser.add_argument("--lr_summary", dest="lr_summary", type=float,
                        default=5e-3, help='learning rate')
    parser.add_argument("--batch_size_per_gpu_summary", dest="batch_size_per_gpu_summary", type=int,
                        default=1, help="batch size per gpu")
    parser.add_argument("--valid_size_per_gpu_summary", dest="valid_size_per_gpu_summary", type=int,
                        default=4, help="valid size per gpu")
    parser.add_argument("--test_size_per_gpu_summary", dest="test_size_per_gpu_summary", type=int,
                        default=4, help="test size per gpu")
    parser.add_argument("--gradient_accumulation_steps_summary", dest="gradient_accumulation_steps_summary", type=int,
                        default=8, help="gradient accumulation steps")
    parser.add_argument("--max_epoch_summary", dest="max_epoch_summary", type=int,
                        default=60, help="max epoch number")
    parser.add_argument("--num_workers_summary", dest="num_workers_summary", type=int,
                        default=0, help="dataloader num_workers")
    parser.add_argument("--weight_decay_summary", dest="weight_decay_summary", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--warmup_steps_summary", dest="warmup_steps_summary", type=float,
                        default=0.01, help="warmup steps")
    parser.add_argument("--max_grad_norm_summary", dest="max_grad_norm_summary", type=float,
                        default=1.0, help="max grad norm")
    parser.add_argument("--eval_step_summary", dest="eval_step_summary", type=int,
                        default=15000, help="how many steps to eval")

    # evaluation
    parser.add_argument("--log_step_pretrain", dest="log_step_pretrain", type=int,
                        default=50, help="how many steps to log")
    parser.add_argument("--log_step_finetune", dest="log_step_finetune", type=int,
                        default=1, help="how many steps to log")
    parser.add_argument("--stemmer", dest="stemmer", type=bool,
                        default=True)
    parser.add_argument("--eval_epoch_0", dest="eval_epoch_0", action='store_true',
                        default=False)
    parser.add_argument("--eval_start_step", dest="eval_start_step", type=int,
                        default=30000, help="how many steps to start eval")
    parser.add_argument("--big_testset", action='store_true', help="whether or not to evaluate using the 2k testset")
    parser.add_argument("--full_testset", action='store_true', help="whether or not to evaluate using the full testset")
    parser.add_argument("--eval_abstractiveness", dest="eval_abstractiveness", type=bool,
                        default=True)

    # generation
    parser.add_argument("--max_length_entity", dest="max_length_entity", type=int,
                        default=128, help="maximum length of the generated entity chain")
    parser.add_argument("--num_beams", dest="num_beams", type=int,
                        default=4, help="number of beams in beam search")
    parser.add_argument("--repetition_penalty", dest="repetition_penalty", type=float,
                        default=2.5, help="repetition penalty")
    parser.add_argument("--length_penalty", dest="length_penalty", type=float,
                        default=1.0, help="length penalty")

    # export
    parser.add_argument("--save_model", dest="save_model", type=bool,
                        default=True, help="whether to save the model or not")

    # Overall pipeline
    ##### pre-training
    parser.add_argument("--pretrain", action='store_true',
                        default=False, help="whether pretrain a T5 tagger")
    parser.add_argument("--build_salient_entities", action='store_true',
                        default=False, help="whether to build the pseudo-labels for pre-training")
    parser.add_argument("--pretraining_train_size", type=int,
                        default=204045, help="pre-training val size")
    parser.add_argument("--pretraining_val_size", type=int,
                        default=1000, help="pre-training train size")
    parser.add_argument("--pretrain_all_weights", action='store_true',
                        default=True, help="whether pretrain a T5 tagger")
    parser.add_argument("--debug_pretrain", action='store_true',
                        default=False, help="whether to just use 100-10 data points")
    ##### fine-tuning
    ######### pre-training
    parser.add_argument("--use_pretrain_ckpt", action='store_false',
                        default=True, help="whether to load the pre-training ckpt before fine-tuning")
    parser.add_argument("--pretrain_ckpt", type=str,
                        default="/home/mathieu/PromptSumm/t5_tagger_pretrained_ckpt/015_n_400k/bestckpt_full_model",
                        help="path to pretrained model")
    parser.add_argument("--pretrain_prompt_ckpt", type=str,
                        default="/home/mathieu/PromptSumm/t5_tagger_pretrained_ckpt/015_n_400k/bestckpt_prompt",
                        help="path to pretrained model prompt")
    ######### entity prompt-tuning
    parser.add_argument("--finetune_entity", action='store_true',
                        default=False, help="whether finetune a tagger using the fewshot summarization data")
    parser.add_argument("--reuse_entity_file", action='store_true',
                        default=False, help="whether to re-use entities already generated")
    ######### summary prompt-tuning
    parser.add_argument("--finetune_summary", action='store_true',
                        default=False, help="whether finetune a tagger using the fewshot summarization data")
    parser.add_argument("--infer_val_entities", action="store_false",
                        default=True, help="whether to run inference with the T5 entity chain prediction on val set")
    parser.add_argument("--use_entity_chain", action='store_false',
                        default=True,
                        help="whether to use the chain of predicted entities or not at all")  # KEEP IT TRUE
    parser.add_argument("--use_t5_tagger", action='store_false',
                        default=True, help="whether use a t5 tagger")
    parser.add_argument("--if_spacy", action='store_false',
                        default=True, help="whether use spacy to supervise the training of T5 tagger")

    args = parser.parse_args()

    args.few_shot = int(args.few_shot)

    dataset_names = ["ccdv/cnn_dailymail", "xsum", "reddit_tifu", "wikihow", "billsum", "samsum", "c4"]
    dataset_versions = ["3.0.0", "default", "long", "all", "default", "samsum", 'en']
    text_keys = ["article", "document", "documents", "text", "text", "dialogue"]
    summary_keys = ["highlights", "summary", "tldr", "headline", "summary", "summary"]
    validation_keys = ["validation", "validation", "", "validation", "test", "validation"]
    test_keys = ["test", "test", "", "test", "test", "test"]
    highlights = [True, False, False, False, False, False, False]
    max_lengths = [512, 512, 512, 512, 1024, 512, 512]
    max_position_embeddings = [1024, 1024, 1024, 1024, 1536 if not ("Finetune" in args.model) else 1024, 1024, 1024]
    max_summary_lengths = [128, 64, 64, 128, 256, 64, 128]
    optimizers = ["adafactor", "adafactor", "adafactor", "adafactor", "adafactor", "adafactor", "adafactor"]
    test_sizes = [11490, 11334, 4222, 5600, 3269, 819]

    idx = dataset_names.index(args.dataset_name)
    if args.dataset_name == 'cnn_dailymail' or args.dataset_name == "ccdv/cnn_dailymail":
        idx = 0
        args.dataset = 'cnndm'
    else:
        args.dataset = args.dataset_name

    args.dataset_version = dataset_versions[idx]
    args.text_key = text_keys[idx]
    args.summary_key = summary_keys[idx]
    args.validation_key = validation_keys[idx]
    args.test_key = test_keys[idx]
    args.highlights = highlights[idx]
    args.max_length = max_lengths[idx]
    args.max_position_embeddings = max_position_embeddings[idx]
    args.max_summary_length = max_summary_lengths[idx]
    args.optimizer_entity = optimizers[idx]
    args.optimizer_summary = optimizers[idx]
    if ("T5" in args.model):
        args.lr_entity = 5e-1
        args.lr_summary = 5e-1
    if ("Finetune" in args.model) or args.tune_weights:
        args.lr_summary = 5e-5
    if args.few_shot == "1":
        args.batch_size_per_gpu_entity = 1
        args.gradient_accumulation_steps_entity = 1
        args.batch_size_per_gpu_summary = 1
        args.gradient_accumulation_steps_summary = 1
    args.max_test_size = min(args.max_test_size, test_sizes[idx])

    args.model_save_folder = f'saved_models/{args.dataset}/{args.few_shot}/'
    if args.model != 'T5MixPrompt':
        if args.model == 'T5SoftPrompt' and args.use_pretrain_ckpt:
            args.model_save_folder += f'{args.model}_v3/'
            logger.info(args.model_save_folder)
        else:
            args.model_save_folder += f'{args.model}/'
    Path(args.model_save_folder).mkdir(parents=True, exist_ok=True)

    return args


def set_logger(args):
    global logger
    if args.local_rank not in [0, -1]:
        logger = Nop()  # for non-main process, set the logger to a dummy no-op object so it won't actually log anything
    else:
        logger = logging.getLogger('root')

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(f"{args.log_dir}/{args.log_name}.log"),
            logging.StreamHandler()
        ]
    )


def main(args):
    device = torch.device("cpu")
    if len(args.cuda) > 0 and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise Exception('device: ', device)
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend="nccl")
    args.device = device
    logger.info(f"device {args.device}")
    args.n_gpu = len(args.cuda.split(","))

    seed_everything(args)

    if args.dataset == 'cnndm':
        allgentasktokens = ["summerizationcnndm"]
        thistaskname = "cnn daily mail"
        thistaskfold = "cnndm"
        args.taskfold = thistaskfold
    else:
        allgentasktokens = ["summerization" + args.dataset]
        thistaskname = args.dataset
        thistaskfold = args.dataset
        args.taskfold = thistaskfold

    # load tokenizer
    if 'Bart' in args.model:
        tokenizer = BartTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)
    elif 'Pegasus' in args.model:
        tokenizer = PegasusTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)
    for gg in range(len(allgentasktokens)):
        gentasktoken = allgentasktokens[gg]
        tokenizer.add_tokens(gentasktoken)
        logger.info('gen token = {} , gen token id = {}'.format(
            gentasktoken, tokenizer.convert_tokens_to_ids(gentasktoken)
        ))
    answertoken = "__ans__"
    special_tokens = {"ans_token": answertoken}
    tokenizer.add_tokens(list(special_tokens.values()))
    tokenizer.add_tokens(['[SEP]'])

    promptnumber = args.prompt_number

    # load datasets
    args.few_shot_save_dir = args.data_dir + args.dataset + "/{}/".format(args.few_shot)
    logger.info(f"Few shot save dir: {args.few_shot_save_dir}")
    dataset_args = [args.dataset_name, args.dataset_version]
    if not os.path.isdir(args.few_shot_save_dir):
        os.makedirs(args.few_shot_save_dir)
    # sample multiple datasets (reproducible with fixed seeds)
    few_shot_seeds = range(args.num_seeds)
    # if files don't exist, subsample
    if (len(os.listdir(args.few_shot_save_dir)) < len(few_shot_seeds)) or not (
    os.path.exists(args.few_shot_save_dir + "seed_0/test.txt")):
        logger.info('subsampling..')
        subsample(dataset_args, few_shot_seeds, args)

    logger.info(f'args.full_testset: {args.full_testset}')
    args.test_file = args.data_dir + args.dataset + '/full_test.txt'
    logger.info(args.test_file)
    args.test_dataset = SummarizationDataset(args.test_file, "valid", args.max_length, tokenizer, allgentasktokens, answertoken, args)
    logger.info(f'args.test_dataset.num_entries: {args.test_dataset.num_entries}')

    args.test_dataset.shuffle_and_subsample()
    logger.info(f'args.test_dataset.num_entries: {args.test_dataset.num_entries}')

    logger.info("\n" + "*" * 50)
    logger.info("3/ Prompt tuning the summarization model...")

    keys = ['best_val_mean_rouge', 'val_rouge1', 'val_rouge2', 'val_rougeL', 'precision', 'recall', 'f1']
    if args.eval_abstractiveness:
        keys += ["new_unigrams", "new_bigrams", "new_trigrams", "new_quadrigrams"]
        keys += ["new_unigrams_target", "new_bigrams_target", "new_trigrams_target", "new_quadrigrams_target"]
    result_dict_total = {}
    for k in keys:
        result_dict_total[k] = []

    seed = 0
    args.model_save_path = args.model_save_folder + f'seed_{seed}/'
    logger.info('args.model_save_path {}'.format(args.model_save_path))
    logger.info('args.save_model {}'.format(args.save_model))

    # base model
    if 'Bart' in args.model:
        basemodel = BartForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
        args.allnumber_path = 'allnumber.pickle'
    elif 'Pegasus' in args.model:
        basemodel = PegasusForConditionalGeneration.from_pretrained(args.model_name, max_position_embeddings=args.max_position_embeddings, cache_dir=args.cache_path)
        args.allnumber_path = 'allnumber.pickle_newforpegasus'
    else:
        basemodel = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
        args.allnumber_path = 'allnumber.pickle'
    logger.info("Finish prepare model and dataset")
    logger.info("Start training")

    if args.model in ['T5Finetune', 'BartFinetune', 'PegasusFinetune']:
        logger.info('\nFinetuning')
        model = ModelFinetune(args, basemodel, tokenizer, args.model)
    elif args.model in ['T5SoftPrompt', 'BartSoftPrompt', 'PegasusSoftPrompt']:
        logger.info('\nSoft prompt tuning')
        model = ModelSoftPrompt(args, basemodel, tokenizer, args.model)
        promptembedding = getpromptembedding(model, tokenizer, promptnumber, thistaskname, args.allnumber_path)
        model.set_prompt_embedding(promptnumber, promptembedding)
    elif args.model in ['T5MixPrompt', 'BartMixPrompt', 'PegasusMixPrompt']:
        logger.info('\nMix prompt tuning')
        model = ModelMixPrompt(args, basemodel, tokenizer, args.model)
        promptembedding = getpromptembedding(model, tokenizer, promptnumber, thistaskname, args.allnumber_path)
        model.set_prompt_embedding(promptnumber, promptembedding)
    else:
        raise Exception('Model not implemented yet')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("The model has {} trainable parameters".format(n_params))

    #####load pre-trained model
    if args.use_pretrain_ckpt and not (args.model in ["T5Finetune", "BartFinetune", "PegasusFinetune"]):
        logger.info("load pre-trained model for summarization")

        # model weights
        ckptsum = torch.load(args.pretrain_ckpt, map_location="cuda:0")
        dicsum = {}
        for x in ckptsum.keys():
            if (args.max_position_embeddings > 1024) and ("embed_positions" in x):
                continue
            if not (x in ["module.promptnumberforsum", "module.promptembeddingforsum"]):
                dicsum[x[7:]] = ckptsum[x]
        if args.max_position_embeddings > 1024:
            dicsum["model.model.encoder.embed_positions.weight"] = basemodel.state_dict()["model.encoder.embed_positions.weight"]
            dicsum["model.model.decoder.embed_positions.weight"] = basemodel.state_dict()["model.decoder.embed_positions.weight"]
        model.load_state_dict(dicsum)

        # just prompt
        ckptsum = torch.load(args.pretrain_prompt_ckpt)
        model.promptnumber = ckptsum["promptnumberforsum"]
        model.promptembedding = nn.parameter.Parameter(ckptsum["promptembeddingforsum"])
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("The model has {} trainable parameters".format(n_params))

    model.eval()
    ####add t5 tagger
    if args.use_t5_tagger and args.model in ["T5MixPrompt", "PegasusMixPrompt"] and args.guidance_mode != "target":
        if args.infer_val_entities:
            ########## predict the validation entity chains with the 1st prompt tuning stage model
            if args.model == "T5MixPrompt":
                entbasemodel = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
                enttokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)
                entmodel = ModelforFinetuneEntity(entbasemodel, enttokenizer, args)
            elif args.model == "PegasusMixPrompt":
                entbasemodel = PegasusForConditionalGeneration.from_pretrained(args.model_name, max_position_embeddings=args.max_position_embeddings, cache_dir=args.cache_path)
                enttokenizer = PegasusTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)
                entmodel = ModelforFinetuneEntity(entbasemodel, enttokenizer, args)

            # Entity model weights

            # from pre-training checkpoint
            if args.use_pretrain_ckpt:
                ckpt = torch.load(args.pretrain_ckpt, map_location="cuda:0")
                dic = {}
                for x in ckpt.keys():
                    if (args.max_position_embeddings > 1024) and ("embed_positions" in x):
                        continue
                    if not (x in ["module.promptnumber", "module.promptembedding", "module.promptnumberforsum", "module.promptembeddingforsum"]):
                        dic[x[7:]] = ckpt[x]
                if args.max_position_embeddings > 1024:
                    dic["model.model.encoder.embed_positions.weight"] = entbasemodel.state_dict()[
                        "model.encoder.embed_positions.weight"]
                    dic["model.model.decoder.embed_positions.weight"] = entbasemodel.state_dict()[
                        "model.decoder.embed_positions.weight"]
                entmodel.load_state_dict(dic)
                logger.info("Loaded the pre-trained ckpt for the entity prediction model!")

            # from entity tuning round
            if args.tune_weights:
                onepath = f'tagger_ckpt/{args.dataset}/{args.few_shot}/seed_{seed}/bestckpt_full_weights'
                if args.use_pretrain_ckpt:
                    onepath += "_from_pretrained"
                oneckpt = torch.load(onepath)
                d = {}
                for k in entmodel.state_dict().keys():
                    d[k] = oneckpt[k]
                entmodel.load_state_dict(d)
                entmodel.promptnumber = oneckpt["promptnumber"]
                entmodel.promptembedding = oneckpt["promptembedding"]
            else:
                onepath = f'tagger_ckpt/{args.dataset}/{args.few_shot}/seed_{seed}/bestckpt_prompt'
                if args.use_pretrain_ckpt:
                    onepath += "_from_pretrained"
                oneckpt = torch.load(onepath)
                entmodel.promptnumber = oneckpt["promptnumber"]
                entmodel.promptembedding = oneckpt["promptembedding"]
            logger.info("Loaded the entity model from: {}".format(onepath))

            n_params = sum(p.numel() for p in entmodel.parameters() if p.requires_grad)
            logger.info("The ent model has {} trainable parameters".format(n_params))
            entmodel.to(args.device)
            logger.info("move to device!")
            model.eval()

            # inferring the entities on the val or test set
            respath = f'tagger_ckpt/{args.dataset}/{args.few_shot}/seed_{seed}/T5valident_{args.test_dataset.num_entries}.pkl'
            if args.big_testset:
                respath = f'tagger_ckpt/{args.dataset}/{args.few_shot}/seed_{seed}/T5_2k_testent_{args.test_dataset.num_entries}.pkl'
            elif args.full_testset:
                respath = f'tagger_ckpt/{args.dataset}/{args.few_shot}/seed_{seed}/T5_full_testent_{args.test_dataset.num_entries}.pkl'
            if args.use_pretrain_ckpt:
                respath = respath[:-4] + "_from_pretrained.pkl"
            if args.tune_weights:
                respath = respath[:-4] + "_full_weights.pkl"
            if not (os.path.isfile(respath) and args.reuse_entity_file):  # to generate, path is there & reuse at the same time
                alldata = args.test_dataset.data
                logger.info(f"test size: {len(alldata)}")
                allresofvalid, allpreds, alllabels = infer_entity_model(alldata, enttokenizer, entmodel, args)
                print(allpreds[:5])
                logger.info(len(allresofvalid))
                with open(respath, "wb") as f:
                    pickle.dump(allresofvalid, f)
                    logger.info("saved the T5 valid entities to: {}".format(respath))
                torch.cuda.empty_cache()
                gc.collect()

            args.test_dataset.set_allent_for_valid(respath)
            logger.info(f'Set valid ents for path: {respath}')

    ########## 2nd prompt tuning stage: summarization
    args.test_dataset.check_entity_keys()

    model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("The model has {} trainable parameters".format(n_params))

    scaler = None
    test_sampler = SequentialSampler(args.test_dataset)
    test_dataloader = get_dataloader(
        tokenizer, args.num_workers_summary, args.test_dataset, args.valid_size_per_gpu_summary, args.max_length,
        args.max_guidance_length, args.test_dataset.tokenizer.pad_token_id, test_sampler, args
    )
    allysrc, allytrue, allypred = doinference(model, test_dataloader, scaler, logger, args)

    d = {}
    d['src'] = []
    d['model'] = []
    model_name = "pegasus"
    if "Mix" in args.model:
        model_name = "promptsum"
    for i in range(len(allysrc)):
        src = allysrc[i]
        pred = allypred[i]
        d['src'].append(src)
        d['model'].append(pred)
    filename = "../human_evaluation/data/{}_{}_{}.pkl".format(args.dataset, model_name, args.max_test_size)
    with open(filename, 'wb') as f:
        pickle.dump(d, f)
        print("exported predictions!")



if __name__ == "__main__":
    args = set_args()
    set_logger(args)
    logger.info(args)
    main(args)


