import argparse
import time
import logging
import pickle5 as pickle

from datasets import load_metric
from transformers import T5Tokenizer

from utils import *
from dataset import *
from guidance import *



parser = argparse.ArgumentParser(description="latentRE")

##### general stuff
parser.add_argument("--cuda", dest="cuda", type=str,
                    default="1", help="gpu id")
parser.add_argument("--seed", dest="seed", type=int,
                    default=42, help="seed for network")

##### data
# 3 datasets: CNN-DM / Reddit TIFU / WikiHow
parser.add_argument("--dataset_name", dest="dataset_name", type=str,
                    default="billsum", help="data name",
                    choices = ["cnn_dailymail", "xsum", "reddit_tifu", "wikihow", "billsum"]) 
parser.add_argument("--dataset_version", dest="dataset_version", type=str,
                    default="default", help="data version",
                    choices = ["3.0.0", "default", "long", "all", "default"]) 
parser.add_argument("--text_key", dest="text_key", type=str,
                    default="text", help="name of the data entry containing the source document",
                    choices = ["article", "document", "documents", "text", "text"]) 
parser.add_argument("--summary_key", dest="summary_key", type=str,
                    default="summary", help="name of the data entry containing the summary",
                    choices = ["highlights", "summary", "tldr", "headline", "summary"])  
parser.add_argument("--validation_key", dest="validation_key", type=str,
                    default="validation", help="name of the dataset field for validation split",
                    choices = ["validation", "validation", "", "validation", "test"])  
parser.add_argument("--test_key", dest="test_key", type=str,
                    default="test", help="name of the dataset field for test split",
                    choices = ["test", "test", "", "test", "test"])  
parser.add_argument("--summary_key", dest="summary_key", type=str,
                    default="summary", help="name of the data entry containing the summary",
                    choices = ["highlights", "summary", "tldr", "headline", "summary"])  
parser.add_argument("--dataset_data_dir", dest="dataset_data_dir", type=str,
                    default=None, help = "folder for WikiHow data") 
parser.add_argument("--dataset_cache_dir", dest="dataset_cache_dir", type=str,
                    default="../../hf_datasets/", help="dataset cache folder")
parser.add_argument("--num_entries", dest="num_entries", type=int,
                    default=42139, help="size of the dataset for Reddit TIFU")

##### model
# input 
parser.add_argument("--max_length", dest="max_length", type=int,
                    default=512, help="max source length")
# base model
parser.add_argument("--model", dest="model", type=str,
                    default="T5Finetune", choices=['T5Prompt', 'T5MixPrompt', 'T5Finetune'])
parser.add_argument("--model_name", dest="model_name", type=str,
                    default="google/t5-v1_1-large", help="{t5-base, google/t5-v1_1-base, google/t5-v1_1-large}")
parser.add_argument("--cache_dir", dest="cache_dir", type=str,
                    default="../../hf_models/t5-v1-large", )
parser.add_argument("--use_lm_adapted", dest="use_lm_adapted", type=bool,
                    default=True, help="whether to use lm_adapted model")
parser.add_argument("--lm_adapted_path", dest="lm_adapted_path", type=str,
                    default="../../lm_adapted_t5model/torch_ckpt/large/pytorch_model.bin",
                    help="The path of lm_adapted model")
parser.add_argument("--if_ckpt_only_model", dest="if_ckpt_only_model", type=bool,
                    default=True, help="If ckpt only contains model. Default: True, only contains model")
# prompt 
parser.add_argument("--prompt_length", dest="prompt_length", type=int,
                    default=100, help="The number of prompt")
parser.add_argument("--prompt_length_task", dest="prompt_length_task", type=int,
                    default=100, help="The number of prompt")
parser.add_argument("--prompt_length_label", dest="prompt_length_label", type=int,
                    default=20, help="The number of prompt")
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
                    default=True, help="whether to filter ents based on the frequency")
parser.add_argument("--build_ents_freq", dest="build_ents_freq", type=bool,
                    default=False, help="whether to build the entities frequency dictionary")
parser.add_argument("--ents_freq_max_len", dest="ents_freq_max_len", type=int,
                    default=10000, help="max number of lines to go through for entity frequency")
parser.add_argument("--min_ents_freq", dest="min_ents_freq", type=int,
                    default=10, help="minimum frequency for the entity")
# 2 - salient sentences
parser.add_argument("--n_top_sents", dest="n_top_sents", type=int,
                    default=2, help="number of salient sentences to use")

parser.add_argument("--ents_stats_max_len", type=int, default=100)

args = parser.parse_args()

# print args
print(args)



def main(args):

    # set seed
    seed_everything(args)

    # tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.model_name,cache_dir=args.cache_dir)

    # data
    dataset_args = [args.dataset_name, args.dataset_version]
    if args.dataset_name in ["cnn_dailymail", "xsum", "wikihow", "billsum"]:
        train_dataset = T5CNNDataset(dataset_args, "train", tokenizer, args)
        valid_dataset = T5CNNDataset(dataset_args, args.validation_key, tokenizer, args)
        test_dataset = T5CNNDataset(dataset_args, args.test_key, tokenizer, args)
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
        train_dataset = T5CNNDataset(dataset_args, train_split, tokenizer, args)
        valid_dataset = T5CNNDataset(dataset_args, valid_split, tokenizer, args)
        test_dataset = T5CNNDataset(dataset_args, test_split, tokenizer, args)


    spacy_nlp = spacy.load("en_core_web_sm")
    # train
    spacy_ents_stats(train_dataset.data, "training", spacy_nlp, args)
    # val
    spacy_ents_stats(valid_dataset.data, "validation", spacy_nlp, args)
    # test    
    spacy_ents_stats(test_dataset.data, "test", spacy_nlp, args)



if __name__ == '__main__':

    main(args)

