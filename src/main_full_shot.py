import os
import pickle
import argparse
import gc
import time
import logging
import spacy
import scipy
from datasets import load_dataset, load_metric
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from transformers import BartConfig, BartTokenizer, BartForConditionalGeneration
from transformers import PegasusConfig, PegasusTokenizer, PegasusForConditionalGeneration
gc.enable()

from hyperparameters import root, cache_path, pretrain_ckpt, pretrain_prompt_ckpt
from utils import *
from dataset.dataset import *
from dataset.dataset_entity import *
from dataset.dataset_summary import DatasetSummary
from engine_pretrain import *
from engine_entity import *
from engine_summary import *
from models.model_summary_finetune import ModelSummaryFinetune
from models.model_summary_soft import ModelSummarySoft
from models.model_summary_mix import ModelSummaryMix



def set_args():
    parser = argparse.ArgumentParser(description="latentRE")

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
                        default= root + "DATASETS/PromptSum/")
    parser.add_argument("--dataset_name", dest="dataset_name", type=str,
                        default="xsum")
    parser.add_argument("--dataset_cache_dir", dest="dataset_cache_dir", type=str,
                        default="../../hf_datasets/", help="dataset cache folder")
    parser.add_argument("--few_shot", dest="few_shot", type=str,
                        default="full", help="full  = full-shot fine-tuning")
    parser.add_argument("--zero_shot", action = 'store_true')
    parser.add_argument("--num_seeds", dest="num_seeds", type=int,
                        default=1, help="number of seeds to sample for training AND validation")
    parser.add_argument("--max_train_size", dest="max_train_size", type=int,
                        default=1000000, help="max training set size")
    parser.add_argument("--max_val_size", dest="max_val_size", type=int,
                        default=2000, help="max validation set size")
    parser.add_argument("--max_test_size", dest="max_test_size", type=int,
                        default=20000, help="max test set size")

    # model
    ##### base model
    parser.add_argument("--model", dest="model", type=str,
                        default="PegasusMixPrompt", choices = ["T5Finetune", "T5SoftPrompt", "T5MixPrompt",
                            "BartFinetune", 'BartSoftPrompt', 'BartMixPrompt',
                            "PegasusFinetune", 'PegasusSoftPrompt', 'PegasusMixPrompt'])
    parser.add_argument("--model_name", dest="model_name", type=str,
                        default="google/pegasus-large", choices = ["t5-base", "google/t5-v1_1-base", "google/t5-v1_1-large", "facebook/bart-base",
                        "facebook/bart-large", "google/pegasus-large"])
    parser.add_argument("--use_lm_adapted", dest="use_lm_adapted", type=int,
                        default=0, help="whether to use lm_adapted model") #if we use bart, then automatically don't use lm_adapted
    parser.add_argument("--lm_adapted_path", dest="lm_adapted_path", type=str,
                        default=root + "lm_adapted_t5model/torch_ckpt/large/pytorch_model.bin",
                        help="The path of lm_adapted model")
    parser.add_argument("--cache_path", dest="cache_path", type=str,
                        default=cache_path,
                        help="The path of huggingface cache")
    # prompt
    parser.add_argument("--concat_mode", dest="concat_mode", type=str,
                        default="concat_right", choices = ["concat_right", "concat_left"])
    parser.add_argument("--prompt_number", dest="prompt_number", type=int,
                        default=100, help="The number of prompt")
    ##### discrete prompt
    parser.add_argument("--guidance_type", dest="guidance_type", type=str,
                        default="ents")
    parser.add_argument("--separator", dest="separator", type=str,
                        default=",", choices=[",", " "])
    parser.add_argument("--guidance_mode", dest="guidance_mode", type=str,
                        default="input", choices=["input", "input_most_frequent", "input_salient_sentences", "input_and_target", "target", "target_unique", 'target_unique_filtered'])
    parser.add_argument("--filter_type", dest="filter_type", type=str,
                        default=None, choices=['PERSON','NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT','WORK_OF_ART','LAW','LANGUAGE','DATE','TIME','PERCENT','MONEY','QUANTITY','ORDINAL','CARDINAL'])
    parser.add_argument("--max_guidance_length", dest="max_guidance_length", type=int,
                        default=100)
    parser.add_argument("--counterfactual_removal", dest="counterfactual_removal", type=bool,
                        default=False, help="whether to use counterfactual removal method during training to enforce causal link")

    # optimization
    parser.add_argument("--adam_epsilon", dest="adam_epsilon", type=float,
                        default = 1e-8, help="adam epsilon")
    ##### pretraining
    parser.add_argument("--optimizer_pretrain", dest="optimizer_pretrain", type=str,
                        default="adafactor", help="optimizer for the pre-training")
    parser.add_argument("--lr_pretrain", dest="lr_pretrain", type=float,
                        default=5e-1, help='learning rate')
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
                        default=10000, help="how many steps to eval")
    ##### entity prompt tuning
    parser.add_argument("--optimizer_entity", dest="optimizer_entity", type=str,
                        default="adafactor", help="optimizer for the entity tuning")
    parser.add_argument("--lr_entity", dest="lr_entity", type=float,
                        default=5e-3, help='learning rate')
    parser.add_argument("--batch_size_per_gpu_entity", dest="batch_size_per_gpu_entity", type=int,
                        default=2, help="batch size per gpu")
    parser.add_argument("--valid_size_per_gpu_entity", dest="valid_size_per_gpu_entity", type=int,
                        default=4, help="valid size per gpu")
    parser.add_argument("--test_size_per_gpu_entity", dest="test_size_per_gpu_entity", type=int,
                        default=4, help="test size per gpu")
    parser.add_argument("--gradient_accumulation_steps_entity", dest="gradient_accumulation_steps_entity", type=int,
                        default=128, help="gradient accumulation steps")
    parser.add_argument("--max_epoch_entity", dest="max_epoch_entity", type=int,
                        default=3, help="max epoch number")
    parser.add_argument("--num_workers_entity", dest="num_workers_entity", type=int,
                        default=4, help="dataloader num_workers")
    parser.add_argument("--weight_decay_entity", dest="weight_decay_entity", type=float,
                        default=0, help="weight decay")
    parser.add_argument("--warmup_steps_entity", dest="warmup_steps_entity", type=float,
                        default=0.01, help="warmup steps")
    parser.add_argument("--max_grad_norm_entity", dest="max_grad_norm_entity", type=float,
                        default=1.0, help="max grad norm")
    parser.add_argument("--eval_step_entity", dest="eval_step_entity", type=int,
                        default=15000, help="how many steps to eval")
    ##### summary prompt tuning
    parser.add_argument("--optimizer_summary", dest="optimizer_summary", type=str,
                        default="adafactor", help="optimizer for the summary fine-tuning")
    parser.add_argument("--train_sample_summary", dest="train_sample_summary", type=bool,
                        default=True, help="dynamic sample or not")
    parser.add_argument("--lr_summary", dest="lr_summary", type=float,
                        default=5e-3, help='learning rate')
    parser.add_argument("--batch_size_per_gpu_summary", dest="batch_size_per_gpu_summary", type=int,
                        default=2, help="batch size per gpu")
    parser.add_argument("--valid_size_per_gpu_summary", dest="valid_size_per_gpu_summary", type=int,
                        default=4, help="valid size per gpu")
    parser.add_argument("--test_size_per_gpu_summary", dest="test_size_per_gpu_summary", type=int,
                        default=4, help="test size per gpu")
    parser.add_argument("--gradient_accumulation_steps_summary", dest="gradient_accumulation_steps_summary", type=int,
                        default=128, help="gradient accumulation steps")
    parser.add_argument("--max_epoch_summary", dest="max_epoch_summary", type=int,
                        default=5, help="max epoch number")
    parser.add_argument("--num_workers_summary", dest="num_workers_summary", type=int,
                        default=0, help="dataloader num_workers")
    parser.add_argument("--weight_decay_summary", dest="weight_decay_summary", type=float,
                        default=0, help="weight decay")
    parser.add_argument("--warmup_steps_summary", dest="warmup_steps_summary", type=float,
                        default=0.01, help="warmup steps")
    parser.add_argument("--max_grad_norm_summary", dest="max_grad_norm_summary", type=float,
                        default=1.0, help="max grad norm")
    parser.add_argument("--eval_step_summary", dest="eval_step_summary", type=int,
                        default=10000, help="how many steps to eval")
    parser.add_argument("--label_smoothing", dest="label_smoothing", type = float,
                        default = 0.0)

    # evaluation
    parser.add_argument("--log_step_pretrain", dest="log_step_pretrain", type=int,
                        default=50, help="how many steps to log")
    parser.add_argument("--log_step_finetune", dest="log_step_finetune", type=int,
                        default=100, help="how many steps to log")
    parser.add_argument("--stemmer", dest="stemmer", type=bool, 
                        default=True)
    parser.add_argument("--full_testset", action='store_true', help="whether or not to evaluate using the full testset")
    parser.add_argument("--eval_abstractiveness", dest="eval_abstractiveness", type=bool,
                        default=True)
    parser.add_argument("--eval_epoch_0", action="store_true", 
                        default=True, help="whether to evaluate before training")
    parser.add_argument("--test_on_val", action="store_true",
                        default=False, help="whether to use the validation for test inference")

    # generation
    parser.add_argument("--max_length_entity", dest="max_length_entity", type=int,
                        default=128, help="maximum length of the generated entity chain")
    parser.add_argument("--num_beams", dest="num_beams", type=int,
                        default=4, help="number of beams in beam search")
    parser.add_argument("--repetition_penalty", dest="repetition_penalty", type=float,
                        default=1.0, help="repetition penalty")
    parser.add_argument("--length_penalty", dest="length_penalty", type=float,
                        default=1.0, help="length penalty")

    # export
    parser.add_argument("--save_model", dest="save_model", type=bool,
                        default=True, help="whether to save the model or not")
    parser.add_argument("--model_save_path", dest="save_model_path", type=str,
                        default="", help="the path where to save the model")

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
                        default=pretrain_ckpt, help="path to pretrained model")
    parser.add_argument("--pretrain_prompt_ckpt", type=str,
                        default=pretrain_prompt_ckpt, help="path to pretrained model prompt")
    ######### entity prompt-tuning
    parser.add_argument("--finetune_entity", action='store_true',
                        default=False, help="whether finetune a T5 tagger using the fewshot summarization data")
    parser.add_argument("--reuse_entity_file", action='store_true',
                        default=False, help="whether to re-use entities already generated")
    ######### summary prompt-tuning
    parser.add_argument("--finetune_summary", action='store_true',
                        default=False, help="whether finetune a T5 tagger using the fewshot summarization data")
    parser.add_argument("--infer_val_entities", action="store_false",
                        default=True, help="whether to run inference with the T5 entity chain prediction on val set")
    parser.add_argument("--use_entity_chain", action='store_false',
                        default=True, help="whether to use the chain of predicted entities or not at all") # KEEP IT TRUE
    parser.add_argument("--use_tagger",  action='store_false',
                        default=True, help="whether use a t5 tagger")
    parser.add_argument("--if_spacy", action='store_false',
                        default=True, help="whether use spacy to supervise the training of T5 tagger")
    ######### inference-time ablations
    parser.add_argument("--no_finetuned_sprompt", action='store_true',
                        default=False, help="whether to run inference with the fine-tuned or just pre-training S-prompt")
    parser.add_argument("--no_sprompt", action='store_true',
                        default=False, help="whether to use the S-prompt at inference")
    parser.add_argument("--no_finetuned_eprompt", action='store_true',
                        default=False, help="whether to run inference with the fine-tuned or just pre-training E-prompt")
    parser.add_argument("--no_entity_chain", action='store_true',
                        default=False, help="whether to use the entity chain at inference")

    args = parser.parse_args()

    if args.full_testset:
        args.eval_epoch_0 = False

    idx = settle_dataset_args(args)

    lrs_finetune = [5e-5, 1e-4, 1e-4, 1e-4, 2e-4, 1e-4]
    lrs_soft = [5, 5e-3, 5e-3, 5e-3, 5e-1, 5e-3]
    max_epoch_entity = [5, 5, 5, 5, 5, 5]
    max_epoch_summary = [5 if not("Finetune" in args.model) else 10, 5 if not("Finetune" in args.model) else 10, 10, 10, 20, 30]
    eval_step_summary = [500, 500, 100, 100, 50, 50]

    if ("T5" in args.model):
        args.lr_entity = 5e-1
        args.lr_summary = 5e-1
    if ("Finetune" in args.model):
        args.lr_summary = lrs_finetune[idx]
    if ("Prompt" in args.model):
        args.lr_entity = lrs_soft[idx]
        args.lr_summary = lrs_soft[idx]
    if args.max_epoch_summary > 0: # meaning, if we are in training mode:
        args.max_epoch_entity = max_epoch_entity[idx]
        args.max_epoch_summary = max_epoch_summary[idx]
    args.eval_step_summary = eval_step_summary[idx]
    args.max_val_size = min(args.max_val_size, val_sizes[idx])
    args.max_test_size = min(args.max_test_size, test_sizes[idx])

    args.model_save_folder = f'summary_ckpt/{args.dataset}/{args.few_shot}/{args.model}/'
    os.makedirs(args.model_save_folder, exist_ok=True)

    return args

def set_logger(args):
    global logger
    if args.local_rank not in [0, -1]:
        logger = Nop() # for non-main process, set the logger to a dummy no-op object so it won't actually log anything
    else:
        logger = logging.getLogger('root')
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt = '%m/%d/%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(f"{args.log_dir}/{args.log_name}.log"),
            logging.StreamHandler()
        ]
    )

def main(args):
    device = torch.device("cpu")
    if len(args.cuda) > 0 and torch.cuda.is_available():
        device = torch.device("cuda")
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend="nccl")
    args.device = device
    logger.info(f"device {args.device}")
    args.n_gpu = len(args.cuda.split(","))

    seed_everything(args)

    # log train
    if args.local_rank in [0, -1]:
        if not os.path.exists("./log"):
            os.mkdir("./log")
        with open("./log/trainner_log", 'a+') as f:
            f.write(str(time.ctime()) + "\n")
            f.write(str(args) + "\n")
            f.write("----------------------------------------------------------------------------\n")
    if args.dataset == 'cnndm':
        allgentasktokens = ["summerizationcnndm"]
        thistaskname = "cnn daily mail"
        thistaskfold = "cnndm"
        args.taskfold = thistaskfold
    else:
        allgentasktokens = ["summerization"+args.dataset]
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
        logger.info(f'gen token = {gentasktoken} , gen token id = {tokenizer.convert_tokens_to_ids(gentasktoken)}')
    answertoken = "__ans__"
    special_tokens = {"ans_token": answertoken}
    tokenizer.add_tokens(list(special_tokens.values()))
    tokenizer.add_tokens(['[SEP]'])

    promptnumber = args.prompt_number

    # load datasets
    args.save_dir = args.data_dir + args.dataset + "/full/"
    os.makedirs(args.save_dir, exist_ok=True)

    dataset_args = [args.dataset_name, args.dataset_version]
    if args.dataset_name == "billsum":
        data = load_dataset(*dataset_args, download_mode="force_redownload", cache_dir=args.dataset_cache_dir)
        test_data = data['test']
        x_data = data['train'].train_test_split(test_size=0.1, shuffle=True)
        train_data = x_data['train']
        valid_data = x_data['test']
    else:
        data = load_dataset(*dataset_args, cache_dir=args.dataset_cache_dir)
        train_data = data['train']
        valid_data = data['validation']
        test_data = data['test']
    logger.info(f"\nTotal size: {len(data)}")

    logger.info(f"\nData size: train: {len(train_data)}, val: {len(valid_data)}, test: {len(test_data)}")
    train_data = train_data[:args.max_train_size]
    valid_data = valid_data.shuffle()
    valid_data = valid_data[:args.max_val_size]
    test_data = test_data[:args.max_test_size]
    logger.info(f"\nFinal data size: train: {len(train_data)}, val: {len(valid_data)}, test: {len(test_data)}")

    train_path = args.save_dir + f'seed_{args.seed}/train.txt'
    valid_path = args.save_dir + f'seed_{args.seed}/valid.txt'
    test_path = args.save_dir + f'seed_{args.seed}/test.txt'
    os.makedirs(args.save_dir + f'seed_{args.seed}', exist_ok=True)
    train_data_new = []
    for i in tqdm(range(len(train_data[list(train_data.keys())[0]]))):
        t = {}
        for k in train_data.keys():
            t[k] = train_data[k][i]
        train_data_new.append(t)
    train_data = train_data_new
    valid_data_new = []
    for i in tqdm(range(len(valid_data[list(valid_data.keys())[0]]))):
        t = {}
        for k in valid_data.keys():
            t[k] = valid_data[k][i]
        valid_data_new.append(t)
    valid_data = valid_data_new
    test_data_new = []
    for i in tqdm(range(len(test_data[list(test_data.keys())[0]]))):
        t = {}
        for k in test_data.keys():
            t[k] = test_data[k][i]
        test_data_new.append(t)
    test_data = test_data_new
    logger.info(f"train/val/test size: {len(train_data)}/{len(valid_data)}/{len(test_data)}")

    convert_data_to_txt(train_data, train_path, args)
    convert_data_to_txt(valid_data, valid_path, args)
    convert_data_to_txt(test_data, test_path, args)

    ########## 1st prompt tuning stage (for entity chain)?
    if args.finetune_entity:
        logger.info("\n"+ "*"*50)
        logger.info("2/ Prompt tuning the tagger for entity chain prediction...")
        # get data
        logger.info("\nprepare data..")
        alltrainfile, allvalidfile, alltestfile = get_data([args.seed], args.save_dir, args)
        # train one T5 tagger for each seed
        logger.info("\nfine-tune...")
        train_tagger_for_all_seeds(alltrainfile, allvalidfile, alltestfile, args)
        return

    ########## 2nd prompt tuning stage (for summarization)?
    if args.finetune_summary:
        logger.info(f'args.full_testset: {args.full_testset}')
        if args.full_testset:
            args.test_file = args.data_dir + args.dataset + '/full_test.txt'
            logger.info(f'full test set file: {args.test_file}')
            # check if we have already generated it
            if not os.path.isfile(args.test_file):
                logger.info('creating')
                subsample_2k_testset(dataset_args, args.test_file, args.seed, args)
            # load
            args.test_dataset = DatasetSummary(args.test_file, "valid", args.max_length, tokenizer, allgentasktokens, answertoken, args, args.seed,
                                            save_path = args.save_dir)
            logger.info(f'args.test_dataset.num_entries: {args.test_dataset.num_entries}')

        logger.info("\n"+ "*"*50)
        logger.info("3/ Prompt tuning the summarization model...")

        # datasets
        train_dataset = DatasetSummary(train_path, "train", args.max_length, tokenizer, allgentasktokens, answertoken, args, args.seed,
                                    save_path = args.save_dir)
        valid_dataset = DatasetSummary(valid_path, "valid", args.max_length, tokenizer, allgentasktokens, answertoken, args, args.seed,
                                    save_path = args.save_dir)
        if args.test_on_val:
            args.test_dataset = valid_dataset

        keys = ['best_val_mean_rouge', 'val_rouge1', 'val_rouge2', 'val_rougeL', "BERTScore", 'precision', 'recall', 'f1']
        if args.eval_abstractiveness:
            keys += ["new_unigrams", "new_bigrams", "new_trigrams", "new_quadrigrams"]
            keys += ["new_unigrams_target", "new_bigrams_target", "new_trigrams_target", "new_quadrigrams_target"]
        result_dict_total = {}
        for k in keys:
            result_dict_total[k] = []

        args.model_save_path = args.model_save_folder + f'seed_{args.seed}/'
        logger.info('args.model_save_path {}'.format(args.model_save_path))
        logger.info('args.save_model {}'.format(args.save_model))

        # base model
        if 'T5' in args.model:
            basemodel = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
            args.allnumber_path = 'support_files/allnumber_t5.pkl'
        elif 'Bart' in args.model:
            basemodel = BartForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
            args.allnumber_path = 'support_files/allnumber_bart.pkl'
        elif 'Pegasus' in args.model:
            basemodel = PegasusForConditionalGeneration.from_pretrained(args.model_name, max_position_embeddings = args.max_position_embeddings, cache_dir=args.cache_path)
            args.allnumber_path = 'support_files/allnumber_pegasus.pkl'
        logger.info("Finish prepare model and dataset")
        logger.info("Start training")

        if args.model in ['T5Finetune', 'BartFinetune', 'PegasusFinetune']:
            logger.info('\nFinetuning')
            model = ModelSummaryFinetune(args, basemodel, tokenizer, args.model)
        elif args.model in ['T5SoftPrompt', 'BartSoftPrompt', 'PegasusSoftPrompt']:
            logger.info('\nSoft prompt tuning')
            model = ModelSummarySoft(args, basemodel, tokenizer, args.model)
            promptembedding = getpromptembedding(model, tokenizer, promptnumber, thistaskname, args.allnumber_path)
            model.set_prompt_embedding(promptnumber, promptembedding)
        elif args.model in ['T5MixPrompt', 'BartMixPrompt', 'PegasusMixPrompt']:
            logger.info('\nMix prompt tuning')
            model = ModelSummaryMix(args, basemodel, tokenizer, args.model)
            promptembedding = getpromptembedding(model, tokenizer, promptnumber, thistaskname, args.allnumber_path)
            model.set_prompt_embedding(promptnumber, promptembedding)
        else:
            raise Exception('Model not implemented yet')

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"The model has {n_params} trainable parameters")

        ##### load pre-trained model
        if args.use_pretrain_ckpt and not(args.model in ["T5Finetune", "BartFinetune", "PegasusFinetune"]):
            logger.info("Load pre-trained model for summarization")

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
            model = model.to(device)

            # just prompt
            ckptsum = torch.load(args.pretrain_prompt_ckpt)
            model.promptnumber = ckptsum["promptnumberforsum"]
            model.promptembedding = nn.parameter.Parameter(ckptsum["promptembeddingforsum"])
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"The model has {n_params} trainable parameters")

        mean_rs_entity = None
        model.eval()
        #### add t5 tagger
        if args.use_tagger and args.model in ["T5MixPrompt", "PegasusMixPrompt"] and args.guidance_mode != "target":
            if args.infer_val_entities:
                ########## predict the validation entity chains with the 1st prompt tuning stage model
                if args.model == "T5MixPrompt":
                    entbasemodel = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir = args.cache_path)
                    enttokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir = args.cache_path)
                    entmodel = ModelEntity(entbasemodel, enttokenizer, args)
                elif args.model == "PegasusMixPrompt":
                    entbasemodel = PegasusForConditionalGeneration.from_pretrained(args.model_name, max_position_embeddings = args.max_position_embeddings, cache_dir = args.cache_path)
                    enttokenizer = PegasusTokenizer.from_pretrained(args.model_name, cache_dir = args.cache_path)
                    entmodel = ModelEntity(entbasemodel, enttokenizer, args)
                logger.info("Loading the pre-trained NER model!")

                # model weights
                if args.use_pretrain_ckpt:
                    ckpt = torch.load(args.pretrain_ckpt, map_location="cuda:0")
                    dic = {}
                    for x in ckpt.keys():
                        if (args.max_position_embeddings > 1024) and ("embed_positions" in x):
                            continue
                        if not (x in ["module.promptnumber", "module.promptembedding", "module.promptnumberforsum", "module.promptembeddingforsum"]):
                            dic[x[7:]] = ckpt[x]
                    if args.max_position_embeddings > 1024:
                        dic["model.model.encoder.embed_positions.weight"] = entbasemodel.state_dict()["model.encoder.embed_positions.weight"]
                        dic["model.model.decoder.embed_positions.weight"] = entbasemodel.state_dict()["model.decoder.embed_positions.weight"]
                    entmodel.load_state_dict(dic)
                    logger.info("Loaded the pre-trained ckpt for the entity prediction model!")

                if not (args.no_finetuned_eprompt):
                    onepath = f'entity_ckpt/{args.dataset}/{args.few_shot}/seed_{args.seed}/bestckpt_prompt_{args.prompt_number}'
                    if args.use_pretrain_ckpt:
                        onepath += "_from_pretrained"
                    oneckpt = torch.load(onepath)
                    entmodel.promptnumber = oneckpt["promptnumber"]
                    entmodel.promptembedding = oneckpt["promptembedding"]
                    logger.info("Loaded the entity model from: {}".format(onepath))
                else:
                    ckpt = torch.load(args.pretrain_prompt_ckpt)
                    entmodel.promptnumber = ckpt["promptnumber"]
                    entmodel.promptembedding = nn.parameter.Parameter(ckpt["promptembedding"])

                n_params = sum(p.numel() for p in entmodel.parameters() if p.requires_grad)
                logger.info(f"The ent model has {n_params} trainable parameters")
                entmodel.to(args.device)
                model.eval()

                respath = f'entity_ckpt/{args.dataset}/{args.few_shot}/seed_{args.seed}/valid_ent.pkl'
                if args.full_testset:
                    respath = f'entity_ckpt/{args.dataset}/{args.few_shot}/seed_{args.seed}/test_ent.pkl'
                if args.use_pretrain_ckpt:
                    respath = respath[:-4] + "_from_pretrained.pkl"
                if not (os.path.isfile(respath) and args.reuse_entity_file):
                    if args.full_testset:
                        alldata = args.test_dataset.data
                        logger.info(f"test size: {len(alldata)}")
                    else:
                        alldata = valid_dataset.data
                        logger.info(f"valid size: {len(alldata)}")
                    allresofvalid, allpreds, alllabels, mean_rs_entity = infer_entity_model(alldata, enttokenizer, entmodel, args)
                    logger.info(len(allresofvalid))
                    with open(respath, "wb") as f:
                        pickle.dump(allresofvalid, f)
                        logger.info("saved the T5 valid entities")
                    torch.cuda.empty_cache()
                    del entmodel, enttokenizer
                    gc.collect()

                if args.full_testset:
                    args.test_dataset.set_allent_for_valid(respath)
                else:
                    valid_dataset.set_allent_for_valid(respath)
                logger.info(f'Set valid ents for path: {respath}')

        ########## 2nd prompt tuning stage: summarization
        model.to(args.device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"The model has {n_params} trainable parameters")

        result_dict = train(tokenizer, model, train_dataset, valid_dataset, logger, args)
        logger.info("Finish training")
        logger.info(f"The model has {n_params} trainable parameters")
        for k in keys:
            result_dict_total[k].append(result_dict[k])

        keys_to_keep = ["mean_rs", "r1s", "r2s", "rls", "bs"]
        d = {}
        for k in keys_to_keep:
            d[k] = result_dict[k]
        is_oracle = bool(args.guidance_mode == "target")
        export_path = f"scores/{args.dataset}/full/prompt_sum_scores_{args.dataset}_pretrained_{args.use_pretrain_ckpt}_oracle_{is_oracle}.pkl"
        if args.no_finetuned_sprompt:
            export_path = export_path[:-4] + "_no_finetuned_sprompt.pkl"
        if args.no_sprompt:
            export_path = export_path[:-4] + "_no_sprompt.pkl"
        if args.no_finetuned_eprompt:
            export_path = export_path[:-4] + "_no_finetuned_eprompt.pkl"
        if args.no_entity_chain:
            export_path = export_path[:-4] + "_no_entity_chain.pkl"
        with open(export_path, "wb") as f:
            pickle.dump(d, f)
            print(f"Saved scores to {export_path}")

        mean_rs_summary = result_dict["mean_rs"]
        if mean_rs_entity != None:
            mean_rs_entity = np.array(mean_rs_entity)
            mean_rs_summary = np.array(mean_rs_summary)
            sort_idx = np.argsort(mean_rs_entity)
            mean_rs_entity = mean_rs_entity[sort_idx]
            mean_rs_summary = mean_rs_summary[sort_idx]
            n_bins = 10
            bin_size = int(len(mean_rs_entity) / n_bins)
            for k in range(n_bins):
                low = k * bin_size
                high = (k + 1) * bin_size
                mean_rs_entity_bin = np.mean(mean_rs_entity[low:high])
                mean_rs_summary_bin = np.mean(mean_rs_summary[low:high])
                print(f"Bin {k}, Mean R entity {mean_rs_entity_bin:.4f} , Mean R summary: {mean_rs_summary_bin:.4f}")
            p, _ = scipy.stats.pearsonr(mean_rs_entity, mean_rs_summary)
            print(f"Pearson correlation: {p:.4f}")

        logger.info('final results:')
        for k in keys:
            easy_results = [f"{x:.2f}" for x in result_dict_total[k]]
            logger.info(f'{k}: {np.mean(result_dict_total[k]):.4f} (all: {easy_results})')

        if args.local_rank != -1:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = set_args()
    set_logger(args)
    logger.info(args)
    main(args)





