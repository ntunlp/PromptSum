import argparse
import time
import logging

import pickle5 as pickle
from datasets import load_metric

from model import *
from model_finetune import T5Finetune
from model_mixture import T5MixPrompt
from dataset import *
from engine import * 
from utils import *



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def get_mix_prompt_embedding(model, tokenizer, task_prompt_length, label_prompt_length):
    def sample_top_k_tokens(topk, t5_embedding):
        with open('allnumber.pickle', 'rb') as fr:
            alltokens = pickle.load(fr)
        sortedalltoken = sorted(alltokens.items(), key=lambda item: item[1], reverse=True)
        top5000 = []
        for one in sortedalltoken:
            if one[0] == 2:
                continue
            else:
                if len(top5000) < 5000:
                    top5000.append(one)
                else:
                    break
        vocab = tokenizer.get_vocab()
        while True:
            topk_emb = []
            touse = random.sample(top5000, topk)
            for tok in touse:
                topk_emb.append(t5_embedding.weight[tok[0]].clone().detach().unsqueeze(0))
            yield torch.cat(topk_emb, 0)

    def get_embs(toks, t5_embedding):
        encoderes = tokenizer.batch_encode_plus([toks], padding=False, truncation=False, return_tensors="pt")
        touse = encoderes["input_ids"].squeeze()[:-1]
        embeddingres = t5_embedding(touse).clone().detach()
        return embeddingres
    t5_embedding = model.model.get_input_embeddings()
    embeddingres = get_embs("summarize this article:", t5_embedding)
    embs_dict = {}
    embs_dict['__task__'] = next(sample_top_k_tokens(task_prompt_length, t5_embedding))
    embs_dict['__task__'][:embeddingres.size(0)] = embeddingres # set meaningful initial tokens 
    return embs_dict

def get_prompt_embedding(model,tokenizer,prompt_length):
    t5_embedding = model.model.get_input_embeddings()
    promptinitembedding = torch.FloatTensor(prompt_length, t5_embedding.weight.size(1))
    startindex = 0
    alllabel = ["summarize this article:"]
    for one in alllabel:
        encoderes = tokenizer.batch_encode_plus([one], padding=False, truncation=False, return_tensors="pt")
        touse = encoderes["input_ids"].squeeze()[:-1]
        embeddingres = t5_embedding(touse).clone().detach()
        if embeddingres.shape[0] > 1:
            embeddingres = torch.mean(embeddingres, 0, keepdim=True)
        promptinitembedding[startindex] = embeddingres
        startindex += 1
    fr = open('allnumber.pickle', 'rb')
    alltokens = pickle.load(fr)
    sortedalltoken = sorted(alltokens.items(), key=lambda item: item[1], reverse=True)
    top5000 = []
    for one in sortedalltoken:
        if one[0] == 2:
            continue
        else:
            if len(top5000) < 5000:
                top5000.append(one)
            else:
                break
    vocab = tokenizer.get_vocab()
    randomtokennum = prompt_length - len(alllabel)
    touse = random.sample(top5000, randomtokennum)
    for one in touse:
        promptinitembedding[startindex] = t5_embedding.weight[one[0]].clone().detach()
        startindex += 1
    return promptinitembedding

def load_prompt(args, model):
    allckpt = torch.load(args.ckpt_path)
    if args.model == 'T5Prompt':
        model.prompt_length = allckpt["prompt_length"]
        model.prompt_embedding = allckpt["prompt_embedding"]
    elif args.model == 'T5MixPrompt':
        model.prompt_dict = allckpt['prompt_dict']
        model.prompt_fix_dict = allckpt['prompt_fix_dict']
        for k, v in model.prompt_fix_dict.items():
            model.prompt_fix_dict[k] = v.to(args.device)
    elif args.model == 'T5Finetune':
        model.load_state_dict(allckpt['t5-base'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentRE")
    parser.add_argument("--cuda", dest="cuda", type=str,
                        default="1", help="gpu id")

    parser.add_argument("--concat_mode", dest="concat_mode", choices=['left_concat', 'right_concat'],
                        default='right_concat', help='append prompt to the left or right')
    parser.add_argument("--order_inv", action="store_true",
                        help="apply order invariance consistency training")  
    parser.add_argument("--subset_inv", action="store_true",
                        help="apply subset invariance consistency training")     
    parser.add_argument("--subset_drop_prob", dest="subset_drop_prob", type=float, default=0.25,
                        help="probability to drop a non ground truth label in prompt during training, only effective when args.subset_inv is turned on")
    parser.add_argument("--continue_learning", action="store_true",
                        help="whether in continual learning setting (assume multiple train/valid files)")

    parser.add_argument("--optimizer", dest="optimizer", choices=['AdamW', 'Adafactor'],
                        default='Adafactor', help='choice of optimizer')
    parser.add_argument("--lr", dest="lr", type=float,
                        default=5e-5, help='learning rate')
    parser.add_argument("--batch_size_per_gpu", dest="batch_size_per_gpu", type=int,
                        default=4, help="batch size per gpu")
    parser.add_argument("--valid_size_per_gpu", dest="valid_size_per_gpu", type=int,
                        default=4, help="valid size per gpu")
    parser.add_argument("--test_size_per_gpu", dest="test_size_per_gpu", type=int,
                        default=4, help="test size per gpu")
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", type=int,
                        default=1, help="gradient accumulation steps")
    parser.add_argument("--max_epoch", dest="max_epoch", type=int,
                        default=5, help="max epoch number")
    parser.add_argument("--num_workers", dest="num_workers", type=int,
                        default=4, help="dataloader num_workers")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--adam_epsilon", dest="adam_epsilon", type=float,
                        default = 1e-8, help="adam epsilon")
    parser.add_argument("--warmup_steps", dest="warmup_steps", type=float,
                        default=0.1, help="warmup steps")
    parser.add_argument("--max_grad_norm", dest="max_grad_norm", type=float,
                        default=1.0, help="max grad norm")

    parser.add_argument("--save_step", dest="save_step", type=int,
                        default=100000, help="step to save")
    parser.add_argument("--log_step", dest="log_step", type=int,
                        default=100, help="how many steps to log")
    parser.add_argument("--eval_step", dest="eval_step", type=int,
                        default=10000000, help="how many steps to eval")

    parser.add_argument("--eval_start_epoch", dest="eval_start_epoch", type=int,
                        default=0, help="after how many epochs to start evaluating")
    parser.add_argument("--eval_epoch", dest="eval_epoch", type=int,
                        default=1, help="how many epochs to eval once")

    parser.add_argument("--save_dir", dest="save_dir", type=str,
                        default="t5_ckpt", help="ckpt dir to save")
    parser.add_argument("--seed", dest="seed", type=int,
                        default=42, help="seed for network")

    parser.add_argument("--model", dest="model", type=str,
                        default="T5Finetune", choices=['T5Prompt', 'T5MixPrompt', 'T5Finetune'])
    parser.add_argument("--model_name", dest="model_name", type=str,
                        default="google/t5-v1_1-base", help="{t5-base,google/t5-v1_1-base}")
    parser.add_argument("--cache_dir", dest="cache_dir", type=str,
                        default="../../hf_models/t5-base-v1", )
    parser.add_argument("--train_file_name", dest="train_file_name", type=str,
                        default="data_conll/", help="train data file path")

    # 3 datasets: CNN-DM / Reddit TIFU / WikiHow
    parser.add_argument("--dataset_name", dest="dataset_name", type=str,
                        default="wikihow", help="data name") # "cnn_dailymail" / "reddit_tifu" / "wikihow"
    parser.add_argument("--dataset_version", dest="dataset_version", type=str,
                        default="all", help="data version") # "3.0.0" / "long" / "all"
    parser.add_argument("--text_key", dest="text_key", type=str,
                        default="text", help="name of the data entry containing the source document") # "article" / "documents" / "text"
    parser.add_argument("--summary_key", dest="summary_key", type=str,
                        default="headline", help="name of the data entry containing the summary") # "highlights" / "tldr" / "headline"
    parser.add_argument("--dataset_data_dir", dest="dataset_data_dir", type=str,
                        default="/data/mathieu/DATASETS/WikiHow/", help = "folder for WikiHow data") # None / None / "/data/mathieu/DATASETS/WikiHow/"
    parser.add_argument("--dataset_cache_dir", dest="dataset_cache_dir", type=str,
                        default="../../hf_datasets/", help="dataset cache folder")
    parser.add_argument("--num_entries", dest="num_entries", type=int,
                    default=42139, help="size of the dataset")

    parser.add_argument("--train_sample", action="store_true",
                        help="dynamic sample or not")
    parser.add_argument("--max_length", dest="max_length", type=int,
                        default=512, help="max source length")
    parser.add_argument("--max_target_length", dest="max_target_length", type=int,
                        default=128, help="max summary length")

    parser.add_argument("--guidance_type", dest="guidance_type", type=str,
                        default="None", help="what kind of guidance. In [ents, sents]")
    parser.add_argument("--guidance_mode", dest="guidance_mode", type=str,
                        default="normal", choices=['oracle', 'normal'], help='if to use oracle guidance')                    
    parser.add_argument("--max_guidance_len", dest="max_guidance_len", type=int,
                        default=40, help="max guidance sequence length")
    # 1 - entities
    parser.add_argument("--ents_stats_max_len", dest="ents_stats_max_len", type=int,
                        default=1000, help="max number of lines to go through for entity stats")
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

    parser.add_argument("--local_rank", dest="local_rank", type=int,
                        default=-1, help="local rank")
    parser.add_argument("--load_ckpt", dest="load_ckpt", type=int,
                        default=0, help="whether load ckpt before training")
    parser.add_argument("--ckpt_path", dest="ckpt_path", type=str,
                        default='', help="The path to prompt ckpt")
    
    parser.add_argument('--save_path', dest="save_path", type=str,
                        default="./reddit_t5_ft_adapted", help="path to save the model")

    parser.add_argument("--use_lm_adapted", dest="use_lm_adapted", type=int,
                        default=1, help="whether to use lm_adapted model")
    parser.add_argument("--lm_adapted_path", dest="lm_adapted_path", type=str,
                        default="../../lm_adapted_t5model/torch_ckpt/base/pytorch_model.bin",
                        help="The path of lm_adapted model")
    parser.add_argument("--prompt_length", dest="prompt_length", type=int,
                        default=100, help="The number of prompt")
    parser.add_argument("--prompt_length_task", dest="prompt_length_task", type=int,
                        default=50, help="The number of prompt")
    parser.add_argument("--prompt_length_label", dest="prompt_length_label", type=int,
                        default=20, help="The number of prompt")
    parser.add_argument("--ifckpt_onlymodel", dest="ifckpt_onlymodel", type=int,
                        default=1, help="If ckpt only contains model. Default: True, only contains model")

    parser.add_argument("--min_ent_freq", dest="min_ent_freq", type=int,
                        default=1, help="Minimum frequency of an entity in the training set")

    args = parser.parse_args()

    # print args
    print(args)
    print ("ckpt path", args.ckpt_path)
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
    #set_seed(args)
    seed_everything(args)
    print(device, os.environ["CUDA_VISIBLE_DEVICES"])

    # log train
    if args.local_rank in [0, -1]:
        if not os.path.exists("./log"):
            os.mkdir("./log")
        with open("./log/trainner_log", 'a+') as f:
            f.write(str(time.ctime()) + "\n")
            f.write(str(args) + "\n")
            f.write("----------------------------------------------------------------------------\n")


    t5model = T5ForConditionalGeneration.from_pretrained(args.model_name,cache_dir=args.cache_dir)
    #print(t5model.get_input_embeddings().weight[2040])
    tokenizer = T5Tokenizer.from_pretrained(args.model_name,cache_dir=args.cache_dir)

    #exit -1
    if args.model == "T5Prompt":
        model = T5Prompt(args,t5model,tokenizer)
        if args.ckpt_path and args.load_ckpt:
            load_prompt(args, model)
        else:
            prompt_length = args.prompt_length
            prompt_embedding = get_prompt_embedding(model, tokenizer, prompt_length)
            model.set_prompt_embedding(prompt_length, prompt_embedding)
        model.to(args.device)
    
    elif args.model == 'T5MixPrompt':
        model = T5MixPrompt(args, t5model, tokenizer)
        if args.ckpt_path and args.load_ckpt:
            load_prompt(args, model)
            model.to(args.device)
        else:
            label_name_embs = get_mix_prompt_embedding(model, tokenizer, args.prompt_length_task, args.prompt_length_label)
            model.to(args.device)
            model.set_prompt_embedding(label_name_embs)

    elif args.model == 'T5Finetune':
        model = T5Finetune(args, t5model, tokenizer)
        model.to(args.device)
        
    else:
        raise Exception("No such model! Please make sure that `model` takes the value in {T5}")
    
    dataset_args = [args.dataset_name, args.dataset_version]
    if args.dataset_name in ["cnn_dailymail", "wikihow"]:
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
        #train_split = train_split[:50]
        #valid_split = valid_split[:10]
        #test_split = test_split[:10]
        train_dataset = T5CNNDataset(dataset_args, args, tokenizer, split=train_split)
        valid_dataset = T5CNNDataset(dataset_args, args, tokenizer, split=valid_split)
        test_dataset = T5CNNDataset(dataset_args, args, tokenizer, split=test_split)

    # Barrier to make sure all process train the model simultaneously.
    if args.local_rank != -1:
        torch.distributed.barrier()

    train(args, model, train_dataset, valid_dataset, test_dataset, logger)

    if args.local_rank in [0, -1]:
        test(args, test_dataset, logger, tokenizer)
    logger.info("Finish training and testing!")

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()

