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
from torch.cuda.amp import autocast as autocast
from torch.utils import data
from torch.utils.data import (
    SequentialSampler, RandomSampler
)
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.grad_scaler import ShardedGradScaler

from model import *
from model_finetune import *
from dataset import *
from utils import *
from engine import *



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
                    default="ccdv/cnn_dailymail")
parser.add_argument("--few_shot", dest="few_shot", type=int,
                    default=10, help="number of data points for training AND validation")

### model
parser.add_argument("--ifckpt_onlymodel", dest="ifckpt_onlymodel", type=int,
                    default=1, help="If ckpt only contains model. Default: True, only contains model")
# input
parser.add_argument("--max_length", dest="max_length", type=int,
                    default=512, help="max sentence length")
# base model
parser.add_argument("--model", dest="model", type=str,
                    default="T5Finetune", help="{T5NER}") # can be T5Summarization, T5Finetune
parser.add_argument("--model_name", dest="model_name", type=str,
                    default="google/t5-v1_1-base", help="{t5-base,google/t5-v1_1-base}")
parser.add_argument("--use_lm_adapted", dest="use_lm_adapted", type=int,
                    default=1, help="whether to use lm_adapted model")
parser.add_argument("--lm_adapted_path", dest="lm_adapted_path", type=str,
                    default="/data/mathieu/lm_adapted_t5model/torch_ckpt/base/pytorch_model.bin",
                    help="The path of lm_adapted model")
parser.add_argument("--cache_path", dest="cache_path", type=str,
                    default="/data/mathieu/hf_models/t5-v1-base/",
                    help="The path of huggingface cache")
parser.add_argument("--dataset_cache_dir", dest="dataset_cache_dir", type=str,
                    default="../../hf_datasets/", help="dataset cache folder")
# prompt
parser.add_argument("--prompt_number", dest="prompt_number", type=int,
                    default=300, help="The number of prompt")

### optimization
parser.add_argument("--train_sample", dest="train_sample", type=bool,
                    default=True, help="dynamic sample or not")
parser.add_argument("--lr", dest="lr", type=float,
                    default=5e-5, help='learning rate')
parser.add_argument("--batch_size_per_gpu", dest="batch_size_per_gpu", type=int,
                    default=1, help="batch size per gpu")
parser.add_argument("--valid_size_per_gpu", dest="valid_size_per_gpu", type=int,
                    default=8, help="valid size per gpu")
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
parser.add_argument("--lm_lambda", dest="lm_lambda", type=float,
                    default=0.0, help='language model loss lambda')

# evaluation
parser.add_argument("--log_step", dest="log_step", type=int,
                    default=1, help="how many steps to log")
parser.add_argument("--eval_step", dest="eval_step", type=int,
                    default=100000, help="how many steps to eval")
parser.add_argument("--stemmer", dest="stemmer", type=bool, 
                    default=True)

# export
parser.add_argument("--save_step", dest="save_step", type=int,
                    default=100000, help="step to save")
parser.add_argument("--save_model", dest="save_model", type=bool,
                    default=False, help="whether to save the model or not")
parser.add_argument("--save_model_path", dest="save_model_path", type=str,
                    default="", help="the path where to save the model")

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
    thistaskname = "cnn daily mail "
    thistaskfold = "cnndm"
    args.taskfold = thistaskfold
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path, local_files_only=True)
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

    promptnumber = args.prompt_number

    # original_way of loading
    # thistrainfilename = args.data_dir + args.dataset + "/{}/seed_0/train.txt".format(args.few_shot)
    # thisvalidfilename = args.data_dir + args.dataset + "/{}/seed_0/valid.txt".format(args.few_shot)
    # # print(thistrainfilename, thisvalidfilename)
    # args.train_file_name = thistrainfilename
    # args.valid_file_name = thisvalidfilename

    # train_dataset = T5SummarizationDataset(args.train_file_name, args.max_length, tokenizer, allgentasktokens, answertoken)
    # valid_dataset = T5SummarizationDataset(args.valid_file_name, args.max_length, tokenizer, allgentasktokens, answertoken)

    # new way of loading
    args.few_shot_save_dir = args.data_dir + args.dataset + "/{}/".format(args.few_shot)
    dataset_args = [args.dataset_name, args.dataset_version]
    if not os.path.isdir(args.few_shot_save_dir):
        os.makedirs(args.few_shot_save_dir)
    # sample multiple datasets (reproducible with fixed seeds)
    few_shot_seeds = [0, 1, 2]
    # if files don't exist, subsample
    if len(os.listdir(args.few_shot_save_dir)) != len(few_shot_seeds):
        logger.info('subsampling..')
        subsample(dataset_args, args, tokenizer, few_shot_seeds)
    # read datasets
    datasets = read_subsampled(args, tokenizer, allgentasktokens, answertoken, few_shot_seeds)
    
    for (train_dataset, valid_dataset) in datasets:
        logger.info("Finish prepare model and dataset")
        logger.info("Start training")


        t5model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
        if args.model == 'T5Finetune':
            print('Finetuning')
            model = T5Finetune(args, t5model, tokenizer)
        elif args.model == 'T5Summarization':
            model = T5forSummarization(args, t5model, tokenizer)
            promptembedding = getpromptembedding(model, tokenizer, promptnumber, thistaskname)
            model.set_prompt_embedding(promptnumber, promptembedding)
        else:
            raise Exception('Model not implemented yet')
        model.to(args.device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("The model has {} trainable parameters".format(n_params))

        result_dict = train(args, model, train_dataset, valid_dataset, logger)
        logger.info("Finish training")
        logger.info("The model has {} trainable parameters".format(n_params))
    print('final results:')
    keys = ['val_rouge1', 'val_rouge2', 'val_rougeL', 'precision', 'recall', 'f1']
    for k in keys:
        print('{}: {}'.format(k, np.mean(result_dict[k])))

    # don't test for now, as it takes too long
    # if args.local_rank in [0, -1]:
    #     logger.info("Start testing")
    #     logger.info("Testing...")
    #     test(args, test_dataset)
    #     logger.info("Finish testing!")

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()



if __name__ == "__main__":
    main(args)





