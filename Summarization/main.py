import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '2'
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
from torch.cuda.amp import autocast as autocast
from torch.utils import data
from torch.utils.data import (
    SequentialSampler, RandomSampler
)
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.grad_scaler import ShardedGradScaler

from utils import *
from dataset_finetune_entity import *
from dataset_finetune_summary import *
from engine_pretrain import *
from engine_finetune_entity import *
from engine_finetune_summary import *

from models_summarization.model_finetune import *
from models_summarization.model_soft import *
from models_summarization.model_mixture import *
from models_summarization.model_mixture_discrete_in_decoder import *
from models_summarization.model_mixture_double_discrete import *


def set_args():
    parser = argparse.ArgumentParser(description="latentRE")

    #root = "/data/qin/"
    #data_root = "/data/ruochen/"
    root = "/data/mathieu/"

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
                        default= root + "DATASETS/PromptSumm/")
    parser.add_argument("--dataset_name", dest="dataset_name", type=str,
                        default="xsum")
    parser.add_argument("--few_shot", dest="few_shot", type=int,
                        default=10, help="number of data points for training AND validation")
    parser.add_argument("--zero_shot", action = 'store_true')
    parser.add_argument("--num_seeds", dest="num_seeds", type=int,
                        default=3, help="number of seeds to sample for training AND validation")

    # model
    ##### input
    parser.add_argument("--max_length", dest="max_length", type=int,
                        default=512, help="max sentence length")
    ##### base model
    parser.add_argument("--model", dest="model", type=str,
                        default="T5MixPrompt", choices = ["T5Finetune", "T5SoftPrompt", "T5MixPrompt",
                            "BartFinetune", 'BartSoftPrompt', 'BartMixPrompt'])
    parser.add_argument("--model_name", dest="model_name", type=str,
                        default="google/t5-v1_1-large", help="{t5-base, google/t5-v1_1-base, facebook/bart-base, facebook/bart-large}")
    parser.add_argument("--use_lm_adapted", dest="use_lm_adapted", type=int,
                        default=1, help="whether to use lm_adapted model") #if we use bart, then automatically don't use lm_adapted
    parser.add_argument("--lm_adapted_path", dest="lm_adapted_path", type=str,
                        default=root + "lm_adapted_t5model/torch_ckpt/large/pytorch_model.bin",
                        help="The path of lm_adapted model")
    parser.add_argument("--cache_path", dest="cache_path", type=str,
                        default=root + "hf_models/t5-v1-large/",
                        help="The path of huggingface cache") # /data/ruochen/hf_models/bart-base for bart
    parser.add_argument("--dataset_cache_dir", dest="dataset_cache_dir", type=str,
                        default="../../hf_datasets/", help="dataset cache folder")
    # prompt
    parser.add_argument("--concat_mode", dest="concat_mode", type=str,
                        default="concat_right", choices = ["concat_right", "concat_left"])
    parser.add_argument("--prompt_number", dest="prompt_number", type=int,
                        default=300, help="The number of prompt")
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
    parser.add_argument("--lr_pretrain", dest="lr_pretrain", type=float,
                        default=5e-1, help='learning rate')
    parser.add_argument("--batch_size_per_gpu_pretrain", dest="batch_size_per_gpu_pretrain", type=int,
                        default=1, help="batch size per gpu")
    parser.add_argument("--valid_size_per_gpu_pretrain", dest="valid_size_per_gpu_pretrain", type=int,
                        default=4, help="valid size per gpu")
    parser.add_argument("--test_size_per_gpu_pretrain", dest="test_size_per_gpu_pretrain", type=int,
                        default=8, help="test size per gpu")
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
                        
    ##### entity prompt tuning
    parser.add_argument("--lr_entity", dest="lr_entity", type=float,
                        default=5e-1, help='learning rate')
    parser.add_argument("--batch_size_per_gpu_entity", dest="batch_size_per_gpu_entity", type=int,
                        default=2, help="batch size per gpu")
    parser.add_argument("--valid_size_per_gpu_entity", dest="valid_size_per_gpu_entity", type=int,
                        default=4, help="valid size per gpu")
    parser.add_argument("--test_size_per_gpu_entity", dest="test_size_per_gpu_entity", type=int,
                        default=8, help="test size per gpu")
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
    ##### summary prompt tuning
    parser.add_argument("--train_sample_summary", dest="train_sample_summary", type=bool,
                        default=True, help="dynamic sample or not")
    parser.add_argument("--lr_summary", dest="lr_summary", type=float,
                        default=5e-1, help='learning rate')
    parser.add_argument("--batch_size_per_gpu_summary", dest="batch_size_per_gpu_summary", type=int,
                        default=1, help="batch size per gpu")
    parser.add_argument("--valid_size_per_gpu_summary", dest="valid_size_per_gpu_summary", type=int,
                        default=4, help="valid size per gpu")
    parser.add_argument("--test_size_per_gpu_summary", dest="test_size_per_gpu_summary", type=int,
                        default=8, help="test size per gpu")
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

    # evaluation
    parser.add_argument("--log_step_pretrain", dest="log_step_pretrain", type=int,
                        default=50, help="how many steps to log")
    parser.add_argument("--log_step_finetune", dest="log_step_finetune", type=int,
                        default=1, help="how many steps to log")
    parser.add_argument("--eval_step", dest="eval_step", type=int,
                        default=15000, help="how many steps to eval")
    parser.add_argument("--stemmer", dest="stemmer", type=bool, 
                        default=True)
    parser.add_argument("--eval_start_step", dest="eval_start_step", type=int,
                        default=30000, help="how many steps to start eval")
    parser.add_argument("--big_testset", dest="big_testset", type=bool,
                        default=False, help="whether or not to evaluate using the 2k testset")                  

    # generation
    parser.add_argument("--num_beams", dest="num_beams", type=int,
                        default=4, help="number of beams in beam search")
    parser.add_argument("--repetition_penalty", dest="repetition_penalty", type=float,
                        default=2.5, help="repetition penalty")
    parser.add_argument("--length_penalty", dest="length_penalty", type=float,
                        default=1.0, help="length penalty")

    # export
    parser.add_argument("--save_model", dest="save_model", type=bool,
                        default=False, help="whether to save the model or not")
    parser.add_argument("--save_model_path", dest="save_model_path", type=str,
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
                        default="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/013_ent_135k/bestckpt_full_model", help="path to pretrained model")
    parser.add_argument("--pretrain_prompt_ckpt", type=str,
                        default="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/013_ent_135k/bestckpt_prompt", help="path to pretrained model prompt")
    ######### entity prompt-tuning
    parser.add_argument("--finetune_entity", action='store_true',
                        default=False, help="whether finetune a T5 tagger using the fewshot summarization data")
    ######### summary prompt-tuning
    parser.add_argument("--finetune_summary", action='store_true',
                        default=False, help="whether finetune a T5 tagger using the fewshot summarization data")
    parser.add_argument("--infer_val_entities", action="store_false",
                        default=True, help="whether to run inference with the T5 entity chain prediction on val set")
    parser.add_argument("--use_entity_chain", action='store_false',
                        default=True, help="whether to use the chain of predicted entities or not at all") # KEEP IT TRUE
    parser.add_argument("--use_t5_tagger",  action='store_false',
                        default=True, help="whether use a t5 tagger")
    parser.add_argument("--if_spacy", action='store_false',
                        default=True, help="whether use spacy to supervise the training of T5 tagger")

    args = parser.parse_args()
    

    dataset_names = ["ccdv/cnn_dailymail", "xsum", "reddit_tifu", "wikihow", "billsum", "samsum","c4"]
    dataset_versions = ["3.0.0", "default", "long", "all", "default", "samsum",'en']
    text_keys = ["article", "document", "documents", "text", "text", "dialogue"]
    summary_keys = ["highlights", "summary", "tldr", "headline", "summary", "summary"]
    validation_keys = ["validation", "validation", "", "validation", "test", "validation"]
    test_keys = ["test", "test", "", "test", "test", "test"]
    highlights = [True, False, False, False, False, False, False]
    max_summary_lengths = [128, 64, 64, 128, 256, 64]

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
    args.max_summary_length = max_summary_lengths[idx]

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

    allgentasktokens = ["summerizationcnndm"]
    thistaskname = "cnn daily mail"
    thistaskfold = "cnndm"
    args.taskfold = thistaskfold

    # load tokenizer 
    if 'Bart' in args.model:
        tokenizer = BartTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)
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
    dataset_args = [args.dataset_name, args.dataset_version]
    if not os.path.isdir(args.few_shot_save_dir):
        os.makedirs(args.few_shot_save_dir)
    # sample multiple datasets (reproducible with fixed seeds)
    few_shot_seeds = range(args.num_seeds)
    # if files don't exist, subsample
    if len(os.listdir(args.few_shot_save_dir)) < len(few_shot_seeds):
        logger.info('subsampling..')
        subsample(dataset_args, args, tokenizer, few_shot_seeds)
    print(args.pretrain_ckpt)
    print(args.pretrain_prompt_ckpt)
    ########## pre-training?
    if args.pretrain:
        logger.info("\n"+ "*"*50)
        logger.info("1/ Pre-training...")
        pretrain_model(dataset_args, args)
        return

    ########## 1st prompt tuning stage (for entity chain)?
    if args.finetune_entity:
        logger.info("\n"+ "*"*50)
        logger.info("2/ Prompt tuning the tagger for entity chain prediction...")
        # get data
        logger.info("\nprepare data..")
        alltrainfile, allvalidfile = get_data(few_shot_seeds, args.few_shot_save_dir, args)
        # train one T5 tagger for each seed
        logger.info("\nfine-tune...")
        train_tagger_for_all_seeds(alltrainfile, allvalidfile, args)
        return

    ########## 2nd prompt tuning stage (for summarization)?
    if args.finetune_summary:
        print('args.big_testset: ', args.big_testset)
        if args.big_testset:
            args.test_file = args.data_dir + args.dataset + '/2k_test.txt'
            # check if we have already generated it
            if not os.path.isfile(args.test_file):
                subsample_2k_testset(dataset_args, args.test_file, args.seed, args)
            args.test_dataset = T5SummarizationDataset(args.test_file, "valid", args.max_length, tokenizer, allgentasktokens, answertoken, args)
      
        logger.info("\n"+ "*"*50)
        logger.info("3/ Prompt tuning the summarization model...")
        # read datasets
        datasets = read_subsampled(args, tokenizer, allgentasktokens, answertoken, few_shot_seeds)
        keys = ['best_val_mean_rouge', 'val_rouge1', 'val_rouge2', 'val_rougeL', 'precision', 'recall', 'f1']
        result_dict_total = {}
        for k in keys:
            result_dict_total[k] = []

        count = 0
        for (train_dataset, valid_dataset, seed) in datasets:
            count += 1
            if count <=0:
                continue
            # base model
            if 'Bart' in args.model:
                basemodel = BartForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
            else:
                basemodel = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
            logger.info("Finish prepare model and dataset")
            logger.info("Start training")

            if args.model == 'T5Finetune':
                logger.info('\nFinetuning')
                model = ModelFinetune(args, basemodel, tokenizer, args.model)
            elif args.model == 'T5SoftPrompt':
                logger.info('\nSoft prompt tuning')
                model = ModelSoftPrompt(args, basemodel, tokenizer, args.model)
                promptembedding = getpromptembedding(model, tokenizer, promptnumber, thistaskname)
                model.set_prompt_embedding(promptnumber, promptembedding)
            elif args.model == 'T5MixPrompt':
                logger.info('\nMix prompt tuning')
                model = ModelMixPrompt(args, basemodel, tokenizer, args.model)
                promptembedding = getpromptembedding(model, tokenizer, promptnumber, thistaskname)
                model.set_prompt_embedding(promptnumber, promptembedding)
            else:
                raise Exception('Model not implemented yet')

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info("The model has {} trainable parameters".format(n_params))

            #####load pre-trained model
            if args.use_pretrain_ckpt and args.model != "T5Finetune":
                logger.info("load pre-trained model for summarization")

                # model weights
                ckptsum = torch.load(args.pretrain_ckpt)
                dicsum = {}
                for x in ckptsum.keys():

                    if not (x in ["module.promptnumberforsum", "module.promptembeddingforsum"]):
                        dicsum[x[7:]] = ckptsum[x]
                model.load_state_dict(dicsum)

                # just prompt
                ckptsum = torch.load(args.pretrain_prompt_ckpt)
                model.promptnumber = ckptsum["promptnumberforsum"]
                model.promptembedding = nn.parameter.Parameter(ckptsum["promptembeddingforsum"])
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info("The model has {} trainable parameters".format(n_params))

            model.eval()
            ####add t5 tagger
            if args.use_t5_tagger and args.model == "T5MixPrompt" and args.guidance_mode != "target":
                if args.infer_val_entities:
                    ########## predict the validation entity chains with the 1st prompt tuning stage model
                    entbasemodel = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir = args.cache_path)
                    enttokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir = args.cache_path)
                    entmodel = T5forFinetuneEntity(entbasemodel, enttokenizer, args)
                    logger.info("Loading the pre-trained NER model!")

                    # model weights
                    ckpt = torch.load(args.pretrain_ckpt)
                    dic = {}
                    for x in ckpt.keys():
                        if not (x in ["module.promptnumber", "module.promptembedding", "module.promptnumberforsum", "module.promptembeddingforsum"]):
                            dic[x[7:]] = ckpt[x]
                    entmodel.load_state_dict(dic)
    
                    # just prompt
                    #onepath = f'{args.few_shot_save_dir}seed_{seed}/data_for_bert_{seed}/tagger/bestckpt_prompt' ####bestckpt_prompt?
                    onepath = f'tagger_ckpt/{args.dataset}/{args.few_shot}/seed_{seed}/bestckpt_prompt'
                    print(onepath)
                    oneckpt = torch.load(onepath)
                    entmodel.promptnumber = oneckpt["promptnumber"]
                    entmodel.promptembedding = oneckpt["promptembedding"]
                
                    n_params = sum(p.numel() for p in entmodel.parameters() if p.requires_grad)
                    logger.info("The ent model has {} trainable parameters".format(n_params))
                    entmodel.to(args.device)
                    logger.info("move to device!")
                    model.eval()

                    alldata = valid_dataset.data
                    #logger.info("valid size: ", len(alldata))
                    print("valid size: ", len(alldata))
                    allresofvalid = {}
                    with torch.no_grad():
                        for step in range(len(alldata)):
                            onedata = alldata[step]
                            inputdata = onedata[0]
                            tempdata = re.sub(' +', ' ', inputdata)
                            inputres = enttokenizer.batch_encode_plus([tempdata], padding=True, max_length=args.max_length, truncation=True, return_tensors="pt")
                            input_ids = inputres["input_ids"].to(args.device)
                            attention_mask = inputres["attention_mask"].to(args.device)
                            input = {"input_ids": input_ids, "attention_mask": attention_mask}
                            tagpreds = entmodel._generative_step_for_tagger(input)
                            allentitylist = tagpreds[0].split(',')
                            if allentitylist == []:
                                allentitylist = ["none"]
                            input_guidance = args.separator.join(list(dict.fromkeys(allentitylist)))
                            allresofvalid[tempdata] = input_guidance
                    logger.info(len(allresofvalid))
                    #respath = f'{args.few_shot_save_dir}seed_{seed}/data_for_bert_{seed}/T5valident.pkl'
                    respath = f'tagger_ckpt/{args.dataset}/{args.few_shot}/seed_{seed}/T5valident.pkl'
                    with open(respath, "wb") as f:
                        pickle.dump(allresofvalid, f)
                        logger.info("saved the T5 valid entities")
                    torch.cuda.empty_cache()
                    del entmodel, enttokenizer
                    gc.collect()
                    valid_dataset.set_allent_for_valid()
            
            ########## 2nd prompt tuning stage: summarization
            model.to(args.device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info("The model has {} trainable parameters".format(n_params))

            result_dict = train(tokenizer, model, train_dataset, valid_dataset, logger, args)
            logger.info("Finish training")
            logger.info("The model has {} trainable parameters".format(n_params))
            for k in keys:
                result_dict_total[k].append(result_dict[k])
        logger.info('final results:')
        for k in keys:
            easy_results = ["{:.2f}".format(x) for x in result_dict_total[k]]
            logger.info('{}: {:.4f} (all: {})'.format(k, np.mean(result_dict_total[k]), easy_results))

        if args.local_rank != -1:
            torch.distributed.destroy_process_group()



if __name__ == "__main__":
    args = set_args()
    set_logger(args)
    logger.info(args)
    main(args)





