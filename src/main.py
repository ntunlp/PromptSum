import os
#os.environ['TRANSFORMERS_CACHE'] = '/export/home/cache/'
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import sys
import argparse
import matplotlib
import pdb
import numpy as np
import time
import random
import time
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
from sklearn import metrics
from torch.utils import data
from collections import Counter
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)
from model import *
from model_finetune import T5Finetune
from model_mixture import T5MixPrompt
from dataset import *
from seqeval.metrics import classification_report,f1_score
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.grad_scaler import ShardedGradScaler
import pickle5 as pickle
from datasets import load_metric

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def seed_everything(args):
    random.seed(args.seed)
    os.environ['PYTHONASSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


tosavepath = "./t5_ckpt"

def save_model(modeltoeval, args, steps):
    if isinstance(modeltoeval, torch.nn.parallel.DistributedDataParallel):
        model = modeltoeval.module
    else:
        model = modeltoeval
    model.eval()
    if not os.path.exists(tosavepath):
            os.mkdir(tosavepath)
    if not os.path.exists(tosavepath + "/" + args.save_dir):
        os.mkdir(tosavepath + "/" + args.save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    if args.model == 'T5Prompt':
        ckpt = {
            "prompt_length": model_to_save.prompt_length,
            "prompt_embedding": model_to_save.prompt_embedding
        }
    elif args.model == 'T5MixPrompt':
        ckpt = {
            "prompt_dict": model_to_save.prompt_dict,
            "prompt_fix_dict": model_to_save.prompt_fix_dict
        }
    elif args.model == 'T5Finetune':
        ckpt = {
            't5-base': model_to_save.model.state_dict(),
        }
    print("about to save")
    torch.save(ckpt, os.path.join(tosavepath + "/" + args.save_dir, "ckptofT5_"+str(steps)))
    print("ckpt saved")

def dooneeval(modeltoeval,valid_dataloader,args,result_dict,optimizer,scaler,i):
    if isinstance(modeltoeval, torch.nn.parallel.DistributedDataParallel):
        model = modeltoeval.module
    else:
        model = modeltoeval
    model.eval()
    allytrue = []
    allypred = []
    with torch.no_grad():
        logger.info(len(valid_dataloader))
        for step, batch in enumerate(valid_dataloader):
            logger.info(step)
            
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device), "input_ents": batch[4].to(args.device)}
            if scaler is not None:
                with autocast():
                    sen, target, preds = model._generative_step(inputs)
                    tarres, predres = target, preds
                    allytrue.extend(tarres)
                    allypred.extend(predres)
            else:
                print(f"eval step: {step}")
                sen, target, preds = model._generative_step(inputs)
                tarres, predres = target, preds
                allytrue.extend(tarres)
                allypred.extend(predres)
    rouge = load_metric('rouge')
    rouge_score = rouge.compute(references=allytrue, predictions=allypred)
    logger.info('----Validation Results Summary----')
    logger.info(len(allypred))
    logger.info(rouge_score)

    result_dict['val_rouge1'].append(rouge_score["rouge1"].mid.fmeasure)
    if result_dict['val_rouge1'][-1] > result_dict['best_val_rouge1']:
        logger.info("{} epoch, best epoch was updated! val_rouge1: {: >4.5f}".format(i,result_dict['val_rouge1'][-1]))
        result_dict["best_val_rouge1"] = result_dict['val_rouge1'][-1]
        if not os.path.exists(tosavepath):
            os.mkdir(tosavepath)
        if not os.path.exists(tosavepath + "/" + args.save_dir):
            os.mkdir(tosavepath + "/" + args.save_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        if args.model == 'T5Prompt':
            ckpt = {
                "prompt_length": model_to_save.prompt_length,
                "prompt_embedding": model_to_save.prompt_embedding
            }
        elif args.model == 'T5MixPrompt':
            ckpt = {
                "prompt_dict": model_to_save.prompt_dict,
                "prompt_fix_dict": model_to_save.prompt_fix_dict
            }
        elif args.model == 'T5Finetune':
            ckpt = {
                't5-base': model_to_save.model.state_dict(),
            }
        print("about to save")
        torch.save(ckpt, os.path.join(tosavepath + "/" + args.save_dir, "ckptofT5_best"))
        print("ckpt saved")

def get_dataloader(num_workers,dataset, batch_size, max_len, max_ent_len, pad_id, sampler):
    collate_fn = SmartBatchingCollate(
        max_length=max_len,
        max_ent_length=max_ent_len,
        pad_token_id=pad_id
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        #shuffle=True, #####?????
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

def test(args, test_dataset):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = get_dataloader(args.num_workers, test_dataset, args.test_size_per_gpu, args.max_length, args.max_ent_len,
                                      test_dataset.tokenizer.pad_token_id,test_sampler)

    t5model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir="/export/home/cache/")
    allckpt = torch.load("./t5_ckpt/" + args.save_dir + "/ckptofT5_best")
    if args.model == 'T5Prompt':
        model = T5Prompt(args, t5model, tokenizer)
        model.prompt_length = allckpt["prompt_length"]
        model.prompt_embedding = allckpt["prompt_embedding"]
    elif args.model == 'T5MixPrompt':
        model = T5MixPrompt(args, t5model, tokenizer)
        model.prompt_dict = allckpt['prompt_dict']
        model.prompt_fix_dict = allckpt['prompt_fix_dict']
    elif args.model == 'T5Finetune':
        model = T5Finetune(args, t5model, tokenizer)
        model_state_dict = {}
        for k,v in allckpt['t5-base'].items():
            model_state_dict['model.'+k] = v
        model.load_state_dict(model_state_dict)
    logger.info("load finished!")

    model.to(args.device)
    model.eval()
    allytrue = []
    allypred = []
    #scaler = ShardedGradScaler()
    scaler = None

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device), "input_ents": batch[4].to(args.device)}
            if scaler is not None:
                with autocast():
                    sen,target,preds = model._generative_step(inputs)
                    tarres, predres = getonebatchresult(sen,target,preds)
                    allytrue.extend(tarres)
                    allypred.extend(predres)
            else:
                sen, target, preds = model._generative_step(inputs)
                tarres, predres = getonebatchresult(sen, target, preds)
                allytrue.extend(tarres)
                allypred.extend(predres)
    report = classification_report(allytrue, allypred, digits=4)
    logger.info("\n%s", report)


def train(args, model, train_dataset, valid_dataset, test_dataset):
    # total step
    step_tot = int(0.5 + train_dataset.num_entries / float(args.gradient_accumulation_steps) / args.batch_size_per_gpu / args.n_gpu) * args.max_epoch

    warmup_steps_total = step_tot * args.warmup_steps
    train_sampler = data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(
        train_dataset)

    valid_sampler = SequentialSampler(valid_dataset)

    train_dataloader = get_dataloader(args.num_workers, train_dataset, args.batch_size_per_gpu, args.max_length, args.max_ent_len,
                                      train_dataset.tokenizer.pad_token_id,train_sampler)
    valid_dataloader = get_dataloader(args.num_workers, valid_dataset, args.valid_size_per_gpu, args.max_length, args.max_ent_len,
                                      valid_dataset.tokenizer.pad_token_id,valid_sampler)


    base_optimizer_arguments = {"lr": args.lr, "clip_threshold": args.max_grad_norm, "decay_rate": -0.8,
                                "weight_decay": args.weight_decay,
                                "scale_parameter": False, "relative_step": False}
    if args.model == 'T5Finetune':
        optimizer = AdamW
        base_optimizer_arguments = {"lr": args.lr, "weight_decay": args.weight_decay}
    else:
        optimizer = Adafactor
    if args.n_gpu > 1: # distributed training
        optimizer = OSS(params=filter(lambda p: p.requires_grad, model.parameters()), optim=optimizer,
                        **base_optimizer_arguments)
        # distributed training
        model = ShardedDDP(model, optimizer)
    else:
        optimizer = optimizer(params=filter(lambda p: p.requires_grad, model.parameters()), **base_optimizer_arguments)
    #import pdb;pdb.set_trace() #len(optimizer.param_groups[0]['params']) #len(optimizer.param_groups)
    model.train()
    #scaler = ShardedGradScaler()
    scheduler = None
    scaler = None


    startepoch = 0
    Best_F1 = 0.0

    logger.info("Begin train...")
    logger.info("We will train model in %d steps" % step_tot)

    result_dict = {
        'epoch': [],
        'val_rouge1': [],
        'best_val_rouge1': Best_F1
    }
    global_step = 0
    model.eval()
    model.train()
    for i in range(startepoch, startepoch + args.max_epoch):
        # if i < 32:
        #     adjusted_evalstep = args.eval_step * 10
        # elif i >= 32:
        adjusted_evalstep = args.eval_step
        #if i > 420:
        #    break

        model.train()
        result_dict['epoch'] = i
        allloss = []
        for step, batch in enumerate(train_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device),
                      "input_ents": batch[4].to(args.device)}

            if scaler is not None:
                with autocast():
                    loss = model(inputs)
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
            else:
                loss = model(inputs)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
            finalloss = loss 
            if scaler is not None:
                scaler.scale(finalloss).backward()
            else:
                finalloss.backward()
            allloss.append(loss.item())
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if scaler is not None:
                    #scaler.unscale_(optimizer)
                    #optimizer.clip_grad_norm(args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    #nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                if scheduler != None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [0, -1] and global_step % args.log_step == 0:
                    #logger.info("step: %d, shcedule: %.3f, loss: %.6f" % (global_step, global_step/step_tot, np.average(allloss)))
                    logger.info("step: %d, schedule: %.3f, loss: %.6f, epoch: %d" % (
                    global_step, global_step / step_tot, np.average(allloss), i))

                if args.local_rank in [0, -1] and global_step % adjusted_evalstep == 0:
                    #####eval
                    #model.eval()
                    #sen, target, preds = model._generative_step(inputs)
                    dooneeval(model,valid_dataloader,args,result_dict,optimizer,scaler,i)
                    #print("only eval every epoch")
                    #print("not eval!!!")
                    model.train()
                    print('back to train')

                if args.local_rank in [0, -1] and global_step % args.save_step == 0:
                    save_model(model, args, global_step)
                    model.train()

        if args.train_sample:
            logger.info("sampling...")
            logger.info("sampled")
    print('finish training')
    if args.local_rank in [0, -1]:
        save_model(model, args, global_step)

# def getalltokennum(tokenizer):
#     file = "../c4/en/dataalltouse.json"
#     f = open(file, 'r')
#     ii = 0
#     while True:
#         line = f.readline().strip()
#         if not line:
#             break
#         content = json.loads(line)
#         text = content['text']
#         text.replace("\n\n\n", " ")
#         text.replace("\n\n", " ")
#         text.replace("\n", " ")
#         encoderes = tokenizer.batch_encode_plus([text], padding=False, truncation=False, return_tensors="pt")
#         touse = encoderes["input_ids"].squeeze()[:-1]
#         for i in range(touse.shape[0]):
#             if touse[i].item() in alltokens:
#                 alltokens[touse[i].item()] += 1
#             else:
#                 alltokens[touse[i].item()] = 1
#         if ii % 100000 == 0:
#             print(ii)
#             with open('allnumber.pickle', 'wb') as handle:
#                 pickle.dump(alltokens, handle, protocol=pickle.HIGHEST_PROTOCOL)
#             #print(alltokens)
#         ii += 1

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
    #print(promptinitembedding)
    startindex = 0
    #print(promptinitembedding.shape)
    #print(t5_embedding.weight.shape)
    # print(tokenizer.get_vocab())
    alllabel = ["summarize this article:"]
    for one in alllabel:
        encoderes = tokenizer.batch_encode_plus([one], padding=False, truncation=False, return_tensors="pt")
        touse = encoderes["input_ids"].squeeze()[:-1]
        # print(touse)
        # print(touse.shape)
        embeddingres = t5_embedding(touse).clone().detach()
        if embeddingres.shape[0] > 1:
            embeddingres = torch.mean(embeddingres, 0, keepdim=True)
        promptinitembedding[startindex] = embeddingres
        startindex += 1
        # print(embeddingres.shape)
    #print(promptinitembedding)
    # alltokens = {}
    fr = open('allnumber.pickle', 'rb')
    alltokens = pickle.load(fr)
    #print(len(alltokens))
    # print(alltokens)
    sortedalltoken = sorted(alltokens.items(), key=lambda item: item[1], reverse=True)
    # print(sortedalltoken)
    top5000 = []
    for one in sortedalltoken:
        if one[0] == 2:
            continue
        else:
            if len(top5000) < 5000:
                top5000.append(one)
            else:
                break
    #print(len(top5000))
    vocab = tokenizer.get_vocab()
    # print(vocab)
    # for one in top5000:
    #    print(one[0],"\t",one[1],"\t",tokenizer.convert_ids_to_tokens(one[0]))
    randomtokennum = prompt_length - len(alllabel)
    touse = random.sample(top5000, randomtokennum)
    #print(touse)
    for one in touse:
        promptinitembedding[startindex] = t5_embedding.weight[one[0]].clone().detach()
        startindex += 1
    # print(startindex)
    # print(promptinitembedding)
    # print(t5_embedding.weight[2040])
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
                        default="4", help="gpu id")

    
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
                        default=16, help="batch size per gpu")
    parser.add_argument("--valid_size_per_gpu", dest="valid_size_per_gpu", type=int,
                        default=24, help="valid size per gpu")
    parser.add_argument("--test_size_per_gpu", dest="test_size_per_gpu", type=int,
                        default=24, help="test size per gpu")
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", type=int,
                        default=1, help="gradient accumulation steps")
    parser.add_argument("--max_epoch", dest="max_epoch", type=int,
                        default=5, help="max epoch number")
    parser.add_argument("--num_workers", dest="num_workers", type=int,
                        default=4, help="dataloader num_workers")

    parser.add_argument("--save_step", dest="save_step", type=int,
                        default=100000, help="step to save")
    parser.add_argument("--log_step", dest="log_step", type=int,
                        default=1, help="how many steps to log")
    parser.add_argument("--eval_step", dest="eval_step", type=int,
                        default=100, help="how many steps to eval")

    parser.add_argument("--save_dir", dest="save_dir", type=str,
                        default="t5_ckpt", help="ckpt dir to save")
    parser.add_argument("--seed", dest="seed", type=int,
                        default=42, help="seed for network")


    parser.add_argument("--model", dest="model", type=str,
                        default="T5MixPrompt", choices=['T5Prompt', 'T5MixPrompt', 'T5Finetune'])
    parser.add_argument("--model_name", dest="model_name", type=str,
                        default="t5-base", help="{t5-base,google/t5-v1_1-base}")
    parser.add_argument("--cache_dir", dest="cache_dir", type=str,
                        default="../../hf_models/t5-base", )
    parser.add_argument("--train_file_name", dest="train_file_name", type=str,
                        default="data_conll/", help="train data file path")
    parser.add_argument("--dataset_name", dest="dataset_name", type=str,
                        default="cnn_dailymail", help="data name")
    parser.add_argument("--dataset_version", dest="dataset_version", type=str,
                        default="3.0.0", help="data version")
    parser.add_argument("--train_sample", action="store_true",
                        help="dynamic sample or not")
    parser.add_argument("--max_length", dest="max_length", type=int,
                        default=128, help="max sentence length")
    parser.add_argument("--max_ent_len", dest="max_ent_len", type=int,
                        default=40, help="max entity sequence length")

    parser.add_argument("--weight_decay", dest="weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--adam_epsilon", dest="adam_epsilon", type=float,
                        default = 1e-8, help="adam epsilon")
    parser.add_argument("--warmup_steps", dest="warmup_steps", type=float,
                        default=0.1, help="warmup steps")
    parser.add_argument("--max_grad_norm", dest="max_grad_norm", type=float,
                        default=1.0, help="max grad norm")

    parser.add_argument("--local_rank", dest="local_rank", type=int,
                        default=-1, help="local rank")
    parser.add_argument("--load_ckpt", dest="load_ckpt", type=int,
                        default=0, help="whether load ckpt before training")
    parser.add_argument("--ckpt_path", dest="ckpt_path", type=str,
                        default='', help="The path to prompt ckpt")
                        
    parser.add_argument("--use_lm_adapted", dest="use_lm_adapted", type=int,
                        default=0, help="whether to use lm_adapted model")
    parser.add_argument("--lm_adapted_path", dest="lm_adapted_path", type=str,
                        default="../t5_ckpt_1_0622_bak/t5_ckpt/ckpt_of_step_100000",
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
    # set cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cpu")
    args.device = device
    args.n_gpu = len(args.cuda.split(","))
    #set_seed(args)
    seed_everything(args)

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

    # logger.info(t5model.config.vocab_size)
    # logger.info(t5model.encoder.embed_tokens.weight.shape)
    # logger.info(t5model.encoder.embed_tokens.weight[32099])
    # logger.info(t5model.encoder.embed_tokens.weight[32100])
    # logger.info(t5model.encoder.embed_tokens.weight[32101])
    # t5model.resize_token_embeddings(len(tokenizer))    ##########???????
    # logger.info(t5model.config.vocab_size)
    # logger.info(t5model.encoder.embed_tokens.weight.shape)
    # print(tokenizer.all_special_tokens)
    # print(tokenizer.get_vocab())

    #exit -1
    if args.model == "T5Prompt":
        model = T5Prompt(args,t5model,tokenizer)
        #print(model.model.get_input_embeddings().weight[2040])
        #prompt_length = 100
        if args.ckpt_path and args.load_ckpt:
            load_prompt(args, model)
        else:
            prompt_length = args.prompt_length
            prompt_embedding = get_prompt_embedding(model, tokenizer, prompt_length)
            #print(promptembedding)
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
    train_dataset = T5CNNDataset(dataset_args, args, tokenizer, split='train[:10%]')
    valid_dataset = T5CNNDataset(dataset_args, args, tokenizer, split='validation[:1%]')
    test_dataset = T5CNNDataset(dataset_args, args, tokenizer, split='test')

    # Barrier to make sure all process train the model simultaneously.
    if args.local_rank != -1:
        torch.distributed.barrier()

    train(args, model, train_dataset, valid_dataset, test_dataset)
    
    if args.local_rank in [0, -1]:
        test(args,test_dataset)
    logger.info("Finish training and testing!")

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()

