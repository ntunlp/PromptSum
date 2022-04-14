import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

from model_finetune import *
from model import *
from model_mixture import *
from model_mixture_discrete_in_decoder import * 
from dataset import *
from utils import *
from engine import *
from T5PromptNER.TrainTaggerforSum import *



parser = argparse.ArgumentParser(description="latentRE")

### general stuff
parser.add_argument("--seed", dest="seed", type=int,
                    default=42, help="seed for network")
parser.add_argument("--cuda", dest="cuda", type=str,
                    default="1", help="gpu id")
parser.add_argument("--local_rank", dest="local_rank", type=int,
                    default=-1, help="local rank")

### data
parser.add_argument("--data_dir", dest="data_dir", type=str,
                    default="/data/mathieu/DATASETS/PromptSumm/")
parser.add_argument("--dataset_name", dest="dataset_name", type=str,
                    default="xsum")
parser.add_argument("--few_shot", dest="few_shot", type=int,
                    default=10, help="number of data points for training AND validation")
parser.add_argument("--zero_shot", action = 'store_true')
parser.add_argument("--num_seeds", dest="num_seeds", type=int,
                    default=3, help="number of seeds to sample for training AND validation")

### model
parser.add_argument("--ifckpt_onlymodel", dest="ifckpt_onlymodel", type=int,
                    default=1, help="If ckpt only contains model. Default: True, only contains model")
# input
parser.add_argument("--max_length", dest="max_length", type=int,
                    default=512, help="max sentence length")
# base model
parser.add_argument("--model", dest="model", type=str,
                    default="T5MixPrompt", choices = ["T5Finetune", "T5SoftPrompt", "T5MixPrompt", "T5MixPromptDID",
                        "BartFinetune", 'BartSoftPrompt', 'BartMixPrompt', 'BartMixPromptUnfreeze'])
parser.add_argument("--model_name", dest="model_name", type=str,
                    default="google/t5-v1_1-large", help="{t5-base, google/t5-v1_1-base, facebook/bart-base, facebook/bart-large}")
parser.add_argument("--use_lm_adapted", dest="use_lm_adapted", type=int,
                    default=1, help="whether to use lm_adapted model") #if we use bart, then automatically don't use lm_adapted
parser.add_argument("--lm_adapted_path", dest="lm_adapted_path", type=str,
                    default="/data/mathieu/lm_adapted_t5model/torch_ckpt/base/pytorch_model.bin",
                    help="The path of lm_adapted model")
parser.add_argument("--cache_path", dest="cache_path", type=str,
                    default="/data/mathieu/hf_models/t5-v1-large/",
                    help="The path of huggingface cache") # /data/ruochen/hf_models/bart-base for bart
parser.add_argument("--dataset_cache_dir", dest="dataset_cache_dir", type=str,
                    default="../../hf_datasets/", help="dataset cache folder")
# prompt
parser.add_argument("--concat_mode", dest="concat_mode", type=str,
                    default="concat_right", choices = ["concat_right", "concat_left"])
parser.add_argument("--prompt_number", dest="prompt_number", type=int,
                    default=300, help="The number of prompt")
# discrete prompt
parser.add_argument("--guidance_type", dest="guidance_type", type=str,
                    default="ents")
parser.add_argument("--separator", dest="separator", type=str,
                    default=",", choices=[",", " "])
parser.add_argument("--guidance_mode", dest="guidance_mode", type=str,
                    default="target", choices=["input", "input_most_frequent", "input_salient_sentences", "input_and_target", "target"])
parser.add_argument("--use_bert_tagger", dest="use_bert_tagger", type=bool,
                    default=False)
parser.add_argument("--max_guidance_length", dest="max_guidance_length", type=int,
                    default=100)
parser.add_argument("--counterfactual_removal", dest="counterfactual_removal", type=bool,
                    default=False, help="whether to use counterfactual removal method during training to enforce causal link")

### optimization
parser.add_argument("--train_sample", dest="train_sample", type=bool,
                    default=True, help="dynamic sample or not")
parser.add_argument("--lr", dest="lr", type=float,
                    default=5e-1, help='learning rate')
parser.add_argument("--batch_size_per_gpu", dest="batch_size_per_gpu", type=int,
                    default=1, help="batch size per gpu")
parser.add_argument("--valid_size_per_gpu", dest="valid_size_per_gpu", type=int,
                    default=4, help="valid size per gpu")
parser.add_argument("--test_size_per_gpu", dest="test_size_per_gpu", type=int,
                    default=8, help="test size per gpu")
parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", type=int,
                    default=8, help="gradient accumulation steps")
parser.add_argument("--max_epoch", dest="max_epoch", type=int,
                    default=30, help="max epoch number")
parser.add_argument("--num_workers", dest="num_workers", type=int,
                    default=0, help="dataloader num_workers")
parser.add_argument("--weight_decay", dest="weight_decay", type=float,
                    default=1e-5, help="weight decay")
parser.add_argument("--adam_epsilon", dest="adam_epsilon", type=float,
                    default = 1e-8, help="adam epsilon")
parser.add_argument("--warmup_steps", dest="warmup_steps", type=float,
                    default=0.01, help="warmup steps")
parser.add_argument("--max_grad_norm", dest="max_grad_norm", type=float,
                    default=1.0, help="max grad norm")

# evaluation
parser.add_argument("--log_step", dest="log_step", type=int,
                    default=1, help="how many steps to log")
parser.add_argument("--eval_step", dest="eval_step", type=int,
                    default=100000, help="how many steps to eval")
parser.add_argument("--stemmer", dest="stemmer", type=bool, 
                    default=True)

##### generation
parser.add_argument("--max_summary_length", dest="max_summary_length", type=int,
                    default=64, help="max summary length")
parser.add_argument("--num_beams", dest="num_beams", type=int,
                    default=4, help="number of beams in beam search")
parser.add_argument("--repetition_penalty", dest="repetition_penalty", type=float,
                    default=2.5, help="repetition penalty")
parser.add_argument("--length_penalty", dest="length_penalty", type=float,
                    default=1.0, help="length penalty")

# export
parser.add_argument("--save_step", dest="save_step", type=int,
                    default=100000, help="step to save")
parser.add_argument("--save_model", dest="save_model", type=bool,
                    default=False, help="whether to save the model or not")
parser.add_argument("--save_model_path", dest="save_model_path", type=str,
                    default="", help="the path where to save the model")

##### T5 tagger
parser.add_argument("--pretrain_t5_tagger", action='store_true',
                    default=True, help="whether pretrain a T5 tagger")
parser.add_argument("--train_t5_tagger", action='store_true',
                    default=True, help="whether finetune a T5 tagger using the fewshot summarization data")
parser.add_argument("--use_t5_tagger",  action='store_true',
                    default=False, help="whether use a t5 tagger")
parser.add_argument("--if_spacy", action='store_true',
                    default=True, help="whether use spacy to supervise the training of T5 tagger")


args = parser.parse_args()

dataset_names = ["ccdv/cnn_dailymail", "xsum", "reddit_tifu", "wikihow", "billsum", "samsum"]
dataset_versions = ["3.0.0", "default", "long", "all", "default", "samsum"]
text_keys = ["article", "document", "documents", "text", "text", "dialogue"]
summary_keys = ["highlights", "summary", "tldr", "headline", "summary", "summary"]
validation_keys = ["validation", "validation", "", "validation", "test", "validation"]
test_keys = ["test", "test", "", "test", "test", "test"]
highlights = [True, False, False, False, False, False]

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

print(args)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

#time.sleep(7000)


def main(args):
    # set cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    args.device = device
    args.n_gpu = len(args.cuda.split(","))
    initialseed = args.seed
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
    special_token_ids = {k: tokenizer.convert_tokens_to_ids(v) for k, v in special_tokens.items()}
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
    # handle few-shot data for BERT tagger
    if args.pretrain_t5_tagger:
        print("\npre-train tagger")
        pretrain_model(dataset_args, args)
    if args.train_t5_tagger:
        print("\ntrain tagger")
        #####get data
        alltrainfile, allvalidfile = get_data(dataset_args, args, few_shot_seeds, tokenizer, args.few_shot_save_dir)
        train_tagger_for_all_seeds(alltrainfile, allvalidfile, args)
        raise Exception
        return
    print(args.use_t5_tagger)
    # read datasets
    datasets = read_subsampled(args, tokenizer, allgentasktokens, answertoken, few_shot_seeds)
    keys = ['best_val_mean_rouge', 'val_rouge1', 'val_rouge2', 'val_rougeL', 'precision', 'recall', 'f1']
    result_dict_total = {}
    for k in keys:
        result_dict_total[k] = []

    count = 0
    for (train_dataset, valid_dataset, seed) in datasets:
        count += 1
        # base model
        if 'Bart' in args.model:
            basemodel = BartForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
        else:
            basemodel = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
        logger.info("Finish prepare model and dataset")
        logger.info("Start training")

        if 'Finetune' in args.model:
            print('\nFinetuning')
            model = ModelFinetune(args, basemodel, tokenizer, args.model)
        elif 'SoftPrompt' in args.model:
            print('\nSoft prompt tuning')
            model = ModelSoftPrompt(args, basemodel, tokenizer, args.model)
            promptembedding = getpromptembedding(model, tokenizer, promptnumber, thistaskname)
            model.set_prompt_embedding(promptnumber, promptembedding)
        elif 'MixPrompt' in args.model and not('DID' in args.model):
            print('\nMix prompt tuning')
            model = ModelMixPrompt(args, basemodel, tokenizer, args.model)
            promptembedding = getpromptembedding(model, tokenizer, promptnumber, thistaskname)
            model.set_prompt_embedding(promptnumber, promptembedding)
        elif 'MixPromptDID' in args.model:
            print('\nMix prompt tuning with discrete prompt in decoder')
            model = ModelMixPromptDID(args, basemodel, tokenizer, args.model)
            promptembedding = getpromptembedding(model, tokenizer, promptnumber, thistaskname)
            model.set_prompt_embedding(promptnumber, promptembedding)            
        else:
            raise Exception('Model not implemented yet')
        ####add t5 tagger
        if args.use_t5_tagger and args.model == "T5MixPrompt":
            ####add tagger embeddings to t5 model
            onepath = f'{args.few_shot_save_dir}seed_{seed}/data_for_bert_{seed}/tagger/bestckpt'
            oneckpt = torch.load(onepath)
            model.set_tagger_embedding(oneckpt["promptembedding"])

        model.to(args.device)
        if args.use_t5_tagger and args.model == "T5MixPrompt":
            valid_dataset.set_tagger_tokenizer(model, tokenizer)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("The model has {} trainable parameters".format(n_params))
        
        result_dict = train(args, tokenizer, model, train_dataset, valid_dataset, logger)
        logger.info("Finish training")
        logger.info("The model has {} trainable parameters".format(n_params))
        for k in keys:
            result_dict_total[k].append(result_dict[k])
    print('final results:')
    for k in keys:
        easy_results = ["{:.2f}".format(x) for x in result_dict_total[k]]
        print('{}: {:.4f} (all: {})'.format(k, np.mean(result_dict_total[k]), easy_results))

    # don't test for now, as it takes too long
    # if args.local_rank in [0, -1]:
    #     logger.info("Start testing")
    #     logger.info("Testing...")
    #     test(args, tokenizer, test_dataset, logger)
    #     logger.info("Finish testing!")

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()



if __name__ == "__main__":
    main(args)





