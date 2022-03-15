import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import time
import logging
import pickle5 as pickle
from datasets import load_metric

from utils import *
from dataset import *
from model import *
from engine import * 



parser = argparse.ArgumentParser(description="latentRE")

##### general stuff
parser.add_argument("--cuda", dest="cuda", type=str,
                    default="0", help="gpu id")
parser.add_argument("--seed", dest="seed", type=int,
                    default=42, help="seed for network")
parser.add_argument("--train", dest="train", type=bool,
                    default=True, help="whether to train or not")
parser.add_argument("--local_rank", dest="local_rank", type=int,
                    default=-1, help="local rank")

##### data
# For the following argument, follow the order "cnndm", "xsum", "reddit", "wikihow", "billsum", "samsum"
parser.add_argument("--dataset_name", dest="dataset_name", type=str,
                    default="ccdv/cnn_dailymail", help="data name",
                    choices = ["ccdv/cnn_dailymail", "xsum", "reddit_tifu", "wikihow", "billsum", "samsum"]) 
parser.add_argument("--dataset_data_dir", dest="dataset_data_dir", type=str,
                    default=None, help = "folder for WikiHow data") 
parser.add_argument("--dataset_cache_dir", dest="dataset_cache_dir", type=str,
                    default="../../hf_datasets/", help="dataset cache folder")
parser.add_argument("--num_entries", dest="num_entries", type=int,
                    default=42139, help="size of the dataset for Reddit TIFU")
parser.add_argument("--few_shot", dest="few_shot", type=int,
                    default=10, help="size of the few_shot dataset, False if want to run on whole dataset")
parser.add_argument("--few_shot_save_dir", dest="few_shot_save_dir", type=str,
                    default='/data/mathieu/DATASETS/PromptSumm/', help="path to save the subsampled datasetss")
parser.add_argument("--run_one_to_debug", dest="run_one_to_debug", type=bool,
                    default=False, help="whether to use only one data sampling seed and one training seed to test, instead of using the averate")
parser.add_argument("--kaggle", dest="kaggle", type=bool,
                    default=True, help="whether to report average validation performance instead of test")

##### model
# input 
parser.add_argument("--max_length", dest="max_length", type=int,
                    default=512, help="max source length")
# base model
parser.add_argument("--model", dest="model", type=str,
                    default="T5Prompt", choices=['T5Prompt', 'T5MixPrompt', 'T5Finetune']) #T5Prompt: with soft prompt tuning
parser.add_argument("--model_name", dest="model_name", type=str,
                    default="google/t5-v1_1-base", help="{t5-base, google/t5-v1_1-base, google/t5-v1_1-large}")
parser.add_argument("--cache_dir", dest="cache_dir", type=str,
                    default="../../hf_models/t5-v1-base", )
parser.add_argument("--use_lm_adapted", dest="use_lm_adapted", type=bool,
                    default=False, help="whether to use lm_adapted model")
parser.add_argument("--lm_adapted_path", dest="lm_adapted_path", type=str,
                    default="/data/ruochen/lm_adapted_t5model/torch_ckpt/base/pytorch_model.bin",
                    help="The path of lm_adapted model")
# prompt 
parser.add_argument("--prompt_length", dest="prompt_length", type=int,
                    default=200, help="The size of the soft prompt")
parser.add_argument("--prompt_length_discrete", dest="prompt_length_discrete", type=int,
                    default=20, help="The size of the discrete prompt")
parser.add_argument("--concat_mode", dest="concat_mode", choices=['left_concat', 'right_concat'],
                    default='right_concat', help='append prompt to the left or right')
# guidance signal
parser.add_argument("--guidance_type", dest="guidance_type", type=str,
                    default="ents", help="What kind of guidance as discrete entities. In [None, ents, sents]")
parser.add_argument("--guidance_mode", dest="guidance_mode", type=str,
                    default="normal", choices=['oracle', 'normal'], help='if to use oracle guidance')
parser.add_argument("--max_guidance_length", dest="max_guidance_length", type=int,
                    default=100, help="max guidance sequence length")
# 1 - entities
parser.add_argument("--filter_ents_freq", dest="filter_ents_freq", type=bool,
                    default=False, help="whether to filter ents based on the frequency")
parser.add_argument("--build_ents_freq", dest="build_ents_freq", type=bool,
                    default=False, help="whether to build the entities frequency dictionary")
parser.add_argument("--ents_freq_max_len", dest="ents_freq_max_len", type=int,
                    default=10000, help="max number of lines to go through for entity frequency")
parser.add_argument("--min_ents_freq", dest="min_ents_freq", type=int,
                    default=10, help="minimum frequency for the entity")
# 2 - salient sentences
parser.add_argument("--n_top_sents", dest="n_top_sents", type=int,
                    default=2, help="number of salient sentences to use")

##### load checkpoint
parser.add_argument("--load_ckpt", dest="load_ckpt", type=bool,
                    default=False, help="whether load ckpt before training")
parser.add_argument("--ckpt_path", dest="ckpt_path", type=str,
                    default='saved_models/cnndm_t5_pt_adapted_mix_freq_thresh/t5_ckpt/ckptofT5_best', help="The path to prompt ckpt")

##### optimization
parser.add_argument("--optimizer", dest="optimizer", choices=['AdamW', 'Adafactor'],
                    default='Adafactor', help='choice of optimizer')
parser.add_argument("--lr", dest="lr", type=float,
                    default=5e-1, help='learning rate') # 5e-5 for FT, 5e-1 for PT
parser.add_argument("--batch_size_per_gpu", dest="batch_size_per_gpu", type=int,
                    default=2, help="batch size per gpu")
parser.add_argument("--valid_size_per_gpu", dest="valid_size_per_gpu", type=int,
                    default=2, help="valid size per gpu")
parser.add_argument("--test_size_per_gpu", dest="test_size_per_gpu", type=int,
                    default=2, help="test size per gpu")
parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", type=int,
                    default=32, help="gradient accumulation steps")
parser.add_argument("--max_epoch", dest="max_epoch", type=int,
                    default=10, help="max epoch number")
parser.add_argument("--num_workers", dest="num_workers", type=int,
                    default=4, help="dataloader num_workers")
parser.add_argument("--weight_decay", dest="weight_decay", type=float,
                    default=1e-5, help="weight decay")
parser.add_argument("--max_grad_norm", dest="max_grad_norm", type=float,
                    default=1.0, help="max grad norm")

##### generation
parser.add_argument("--max_summary_length", dest="max_summary_length", type=int,
                    default=128, help="max summary length")
parser.add_argument("--num_beams", dest="num_beams", type=int,
                    default=4, help="number of beams in beam search")
parser.add_argument("--repetition_penalty", dest="repetition_penalty", type=float,
                    default=2.5, help="repetition penalty")
parser.add_argument("--length_penalty", dest="length_penalty", type=float,
                    default=1.0, help="length penalty")

##### evaluation
parser.add_argument("--log_step", dest="log_step", type=int,
                    default=100, help="how many steps to log")
parser.add_argument("--eval_step", dest="eval_step", type=int,
                    default=10000000, help="how many steps to eval")
parser.add_argument("--eval_start_epoch", dest="eval_start_epoch", type=int,
                    default=0, help="after how many epochs to start evaluating")
parser.add_argument("--eval_epoch", dest="eval_epoch", type=int,
                    default=1, help="how many epochs to eval once")
parser.add_argument("--save_step", dest="save_step", type=int,
                    default=10000000, help="step to save")
parser.add_argument("--save_dir", dest="save_dir", type=str,
                    default="t5_ckpt", help="ckpt dir to save")
parser.add_argument("--display_preds", dest="display_preds", type=bool,
                    default=False, help="whether to display predictions during training")

args = parser.parse_args()

#Set dataset related documents according to args.dataset_name
dataset_names = ["ccdv/cnn_dailymail", "xsum", "reddit_tifu", "wikihow", "billsum", "samsum"]
dataset_versions = ["3.0.0", "default", "long", "all", "default", "samsum"]
text_keys = ["article", "document", "documents", "text", "text", "dialogue"]
summary_keys = ["highlights", "summary", "tldr", "headline", "summary", "summary"]
validation_keys = ["validation", "validation", "", "validation", "test", "validation"]
test_keys = ["test", "test", "", "test", "test", "test"]

idx = dataset_names.index(args.dataset_name)
if args.dataset_name == 'cnn_dailymail' or args.dataset_name == "ccdv/cnn_dailymail":
    idx = 0
    save_name = 'cnndm'
else:
    save_name = args.dataset_name

args.dataset_version = dataset_versions[idx]
args.text_key = text_keys[idx]
args.summary_key = summary_keys[idx]
args.validation_key = validation_keys[idx]
args.test_key = test_keys[idx]
args.save_path = f"../../saved_models/{save_name}/t5_base_{args.model}/"
os.makedirs(args.save_path, exist_ok=True)
if args.few_shot:
    args.few_shot_save_dir = args.few_shot_save_dir + f'{save_name}/{args.few_shot}/'
    os.makedirs(args.few_shot_save_dir, exist_ok=True)

# print args
print(args)
print ("ckpt path", args.ckpt_path)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def main(args):

    # set seed
    seed_everything(args)

    # set cuda
    if torch.cuda.is_available():
        if args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = len(args.cuda.split(","))
    else:
        device = torch.device("cpu")
        args.n_gpu = 1
    print("Device: {}".format(device))
    args.device = device

    # log train
    if args.local_rank in [0, -1]:
        if not os.path.exists("./log"):
            os.mkdir("./log")
        with open("./log/trainner_log", 'a+') as f:
            f.write(str(time.ctime()) + "\n")
            f.write(str(args) + "\n")
            f.write("----------------------------------------------------------------------------\n")

    # data
    dataset_args = [args.dataset_name, args.dataset_version]
    if args.few_shot != False:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name,cache_dir=args.cache_dir)
        if args.run_one_to_debug:
            few_shot_seeds = [0]
            training_seeds = [10]
        else:
            few_shot_seeds = [0, 1, 2, 3, 4] #[0, 1, 2, 3, 4]
            training_seeds = [10, 11]
        test_dataset = T5CNNDataset(dataset_args, args, tokenizer, split='test')
        if len(os.listdir(args.few_shot_save_dir)) != len(few_shot_seeds)*2:
            logger.info('subsampling..')
            subsample(dataset_args, args, tokenizer, few_shot_seeds, args.few_shot_save_dir)
        # read in saved few-shot datasets
        datasets = read_subsampled(dataset_args, args, few_shot_seeds, tokenizer, args.few_shot_save_dir)
        if args.kaggle:
            metrics = ['best_val_mean_rouge', 'val_mean_rouge', 'val_rouge1', 'val_rouge2', 'val_rougeL', 'precision', 'recall', 'f1']
        else:
            metrics = ['test_rouge1', 'test_rouge2', 'test_rougeL', 'precision', 'recall', 'f1']
        result_dict = {}
        for m in metrics:
            result_dict[m] = []
        # for each subsampled dataset
        count = 0
        for (train_dataset, valid_dataset) in datasets:
            logger.info(">"*50 + " Dataset {} / {}".format(count + 1, len(datasets)))
            # for each training seed
            for t_seed in training_seeds:
                logger.info(">"*20 + f' Training few shot model with training_seed {t_seed}')
                args.seed = t_seed
                seed_everything(args)
                
                # load model
                model, tokenizer = load_model(args)
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info("The model has {} trainable parameters".format(n_params))

                # Barrier to make sure all process train the model simultaneously.
                if args.local_rank != -1:
                    torch.distributed.barrier()

                if args.train:
                    rd = train(args, model, train_dataset, valid_dataset, test_dataset, logger)

                if args.local_rank in [0, -1]:
                    if not args.kaggle:
                        # to test on full dataset
                        rd = test(args, test_dataset, tokenizer, logger)
                    # else Kaggle way: to report main performance on validation set

                for m in metrics:
                    result_dict[m].append(rd[m])
                    
                logger.info("Finish training and testing!")

                if args.local_rank != -1:
                    torch.distributed.destroy_process_group()
            count += 1
        for m in metrics:
            result_dict[m] = np.mean(np.array(result_dict[m]))
        # report average
        logger.info(f'Final test average: {result_dict}')
    else:
        model, tokenizer = load_model(args)
        if args.dataset_name in dataset_names:
            train_dataset = T5CNNDataset(dataset_args, args, tokenizer, split='train')
            valid_dataset = T5CNNDataset(dataset_args, args, tokenizer, split='validation')
            test_dataset = T5CNNDataset(dataset_args, args, tokenizer, split='test')
        else:
            # build a train:valid:test split
            num_entries = args.num_entries
            print("Total # of data points: {}".format(num_entries))
            idx = np.random.permutation(num_entries)
            thresh = int(0.1 * num_entries)
            # call splits based on their indices
            train_split = list(idx[0:(8 * thresh)])
            valid_split = list(idx[(8 * thresh):(9 * thresh)])
            test_split = list(idx[(9 * thresh):])
            train_split = train_split[:50]
            valid_split = valid_split[:10]
            test_split = test_split[:10]
            train_dataset = T5CNNDataset(dataset_args, args, tokenizer, split=train_split)
            valid_dataset = T5CNNDataset(dataset_args, args, tokenizer, split=valid_split)
            test_dataset = T5CNNDataset(dataset_args, args, tokenizer, split=test_split)

        # Barrier to make sure all process train the model simultaneously.
        if args.local_rank != -1:
            torch.distributed.barrier()

        if args.train:
            train(args, model, train_dataset, valid_dataset, test_dataset, logger)

        if args.local_rank in [0, -1]:
            test(args, test_dataset, tokenizer, logger)
        logger.info("Finish training and testing!")

        if args.local_rank != -1:
            torch.distributed.destroy_process_group()



if __name__ == '__main__':

    main(args)

