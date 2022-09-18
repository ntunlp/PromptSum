import argparse
import time
import logging
import pickle5 as pickle
import spacy

from tqdm import tqdm
from datasets import load_metric
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, PegasusConfig

from utils import *
from dataset import *



parser = argparse.ArgumentParser(description="latentRE")

##### general stuff
parser.add_argument("--cuda", dest="cuda", type=str,
                    default="1", help="gpu id")
parser.add_argument("--seed", dest="seed", type=int,
                    default=42, help="seed for network")

##### data
# For the following argument, follow the order "cnndm", "xsum", "reddit", "wikihow", "billsum", "samsum"
parser.add_argument("--dataset_name", dest="dataset_name", type=str,
                    default="samsum", help="data name",
                    choices = ["cnn_dailymail", "xsum", "reddit_tifu", "wikihow", "billsum", "samsum"])
parser.add_argument("--dataset_data_dir", dest="dataset_data_dir", type=str,
                    default=None, help = "folder for WikiHow data") 
parser.add_argument("--dataset_cache_dir", dest="dataset_cache_dir", type=str,
                    default="../../hf_datasets/", help="dataset cache folder")

##### model
# input 
parser.add_argument("--max_length", dest="max_length", type=int,
                    default=512, help="max source length")
# base model
parser.add_argument("--model", dest="model", type=str,
                    default="T5Finetune", choices=['T5Prompt', 'T5MixPrompt', 'T5Finetune'])
parser.add_argument("--model_name", dest="model_name", type=str,
                    default="google/pegasus-large")
parser.add_argument("--cache_dir", dest="cache_dir", type=str,
                    default="../../hf_models/pegasus-large", )
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

parser.add_argument("--ents_stats_max_len", type=int, default=1000)

args = parser.parse_args()

dataset_names = ["ccdv/cnn_dailymail", "xsum", "reddit_tifu", "wikihow", "billsum", "samsum"]
dataset_versions = ["3.0.0", "default", "long", "all", "default", "samsum"]
text_keys = ["article", "document", "documents", "text", "text", "dialogue"]
summary_keys = ["highlights", "summary", "tldr", "headline", "summary", "summary"]
validation_keys = ["validation", "validation", "", "validation", "test", "validation"]
test_keys = ["test", "test", "", "test", "test", "test"]

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

# print args
print(args)



def main(args):
    # set seed
    seed_everything(args)

    if 'Bart' in args.model:
        tokenizer = BartTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)
    elif 'Pegasus' in args.model:
        tokenizer = PegasusTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)

    # data
    spacy_nlp = spacy.load("en_core_web_sm")
    sets = ["val", "test", "train"]
    for set in sets:
        path = "/home/mathieu/DATASETS/PromptSumm/{}/full/seed_42/{}.txt".format(args.dataset, set)
        print(path)
        count = 0
        source_ents, summary_ents = 0, 0
        with open(path, 'r') as f:
            for l in tqdm(f.readlines()):
                data = l.split("\t")
                source = data[0]
                ents = spacy_nlp(source).ents
                allents = [ent.text for ent in ents]
                source_ents += len(allents)
                summary = data[1]
                ents = spacy_nlp(summary).ents
                allents = [ent.text for ent in ents]
                summary_ents += len(allents)

                count += 1
                if count >= 1000:
                    break
        source_ents /= count
        summary_ents /= count
        print("Average # entities in the source: {:.4f}".format(source_ents))
        print("Average # entities in the summary: {:.4f}".format(summary_ents))




if __name__ == '__main__':

    main(args)

