import pickle
import argparse
import gc
import spacy
import time
import logging

gc.enable()

from datasets import load_metric
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

from utils import *
from dataset_pretrain import *
from dataset_finetune import *
from models_summarization.model_soft import *
from model_finetune_entity import *
from engine_pretrain import *



def train_tagger_for_all_seeds(alltrainfile, allvalidfile, args):
    all_f1s, all_meanRs = [], []
    for i in range(len(alltrainfile)):
        result_dict = train_tagger_for_one_seed(alltrainfile[i], allvalidfile[i], args)
        f1 = result_dict["best_val_F1"]
        meanR = result_dict["best_val_meanR"]
        all_f1s.append(f1)
        all_meanRs.append(meanR)
    f1 = np.mean(all_f1s)
    clean_f1s = ["{:.4f}".format(x) for x in all_f1s]
    print("Mean F1: {:.4f} (over all seeds: {})".format(f1, clean_f1s))
    meanR = np.mean(all_meanRs)
    clean_meanRs = ["{:.4f}".format(x) for x in all_meanRs]
    print("Mean mean ROUGE: {:.4f} (over all seeds: {})".format(meanR, clean_meanRs))


def train_tagger_for_one_seed(trainfile, validfile, args):
    result_dict = finetune_model_tagger(trainfile, validfile, args)

    return result_dict


def finetune_model_tagger(trainfile, validfile, args):
    print("Fine-tuning entity tagger...")
    print(trainfile, validfile)

    ###train
    gradient_accumulation_steps = 2
    train_batch_size = 2
    eval_batch_size = 4
    num_train_epochs = 60  ### epochs for training tagger
    learning_rate = 5e-1
    weight_decay = 1e-5
    max_seq_length = 512
    num_workers = 4
    max_grad_norm = 1.0
    log_step = 1
    model_name = "google/t5-v1_1-large"

    t5model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir="/data/qin/hf_models/t5-v1-large/")
    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir="/data/qin/hf_models/t5-v1-large/")
    model = T5forFinetuneEntity(t5model, tokenizer, args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("The model has {} trainable parameters".format(n_params))

    ##### load from conll ckpt, from pre-training ckpt, or simply initializing?
    if args.use_pretrain_ckpt:
        print("Loading the pre-trained NER model!")

        # full model
        ckpt = torch.load("t5_tagger_pretrained_ckpt/bestckpt_full_model_39k")
        dic = {}
        for x in ckpt.keys():
            if not (x in ["promptnumber", "promptembedding", "promptnumberforsum", "promptembeddingforsum"]):
                dic[x] = ckpt[x]
        model.load_state_dict(dic)

        # just prompt
        ckpt = torch.load("t5_tagger_pretrained_ckpt/bestckpt_prompt_39k")
        model.promptnumber = ckpt["promptnumber"]
        model.promptembedding = ckpt["promptembedding"]
    else:
        ifuseconll = True
        if ifuseconll:
            print("Loading the the CONLL NER model!")
            allckpt = torch.load("./T5PromptNER/bestckpt")
            model.promptnumber = allckpt["promptnumber"]
            model.promptembedding = allckpt["promptembedding"]
        else:
            print("Initializing from scratch!")
            promptnumber = 300
            taskname = "name entity recognition"
            promptembedding = getpromptembedding(model, tokenizer, promptnumber, taskname)
            print("prompt", promptembedding.shape)
            model.set_prompt_embedding(promptnumber, promptembedding)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("The model has {} trainable parameters".format(n_params))
    model.to(args.device)

    train_dataset = T5DatasetPretrainConll(trainfile, max_seq_length, tokenizer)
    valid_dataset = T5DatasetPretrainConll(validfile, max_seq_length, tokenizer)

    if args.local_rank != -1:
        torch.distributed.barrier()

    train_sampler = data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1 \
        else data.RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)

    train_dataloader = get_dataloader_tag(num_workers, train_dataset, train_batch_size, max_seq_length,
                                          train_dataset.tokenizer.pad_token_id, train_sampler)
    valid_dataloader = get_dataloader_tag(num_workers, valid_dataset, eval_batch_size, max_seq_length,
                                          valid_dataset.tokenizer.pad_token_id, valid_sampler)

    #####the path of tuned model
    pos = trainfile.find("docwithlabel_train")
    foldername = trainfile[0:pos]
    print(foldername)
    output_dir = foldername + "tagger"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_optimizer_arguments = {
        "lr": learning_rate,
        "clip_threshold": max_grad_norm,
        "decay_rate": -0.8,
        "weight_decay": weight_decay,
        "scale_parameter": False,
        "relative_step": False
    }
    optimizer = Adafactor(params=filter(lambda p: p.requires_grad, model.parameters()), **base_optimizer_arguments)
    # distributed training
    model.train()

    startepoch = 0
    Best_F1 = 0.0
    Best_val_meanR = 0.0

    logger.info("Begin train...")

    result_dict = {
        'epoch': [],
        'val_F1': [],
        'best_val_F1': Best_F1,
        'val_r1': [],
        'val_r2': [],
        'val_rl': [],
        'best_val_meanR': Best_val_meanR
    }
    global_step = 0
    for i in range(startepoch, startepoch + num_train_epochs):
        thisevalstep = 1000000
        logger.info(i)
        model.train()
        result_dict['epoch'] = i
        allloss = []
        for step, batch in enumerate(train_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
            loss = model(inputs)
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            allloss.append(loss.item())

            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [0, -1] and global_step % log_step == 0:
                    logger.info("step: %d,  loss: %.6f" % (global_step, np.average(allloss)))

                if args.local_rank in [0, -1] and global_step % thisevalstep == 0:
                    print("only eval after every epoch")
                    model.train()

        logger.info("finish one epoch")
        if args.local_rank in [0, -1]:
            dooneeval(model, valid_dataloader, result_dict, i, output_dir, args)

    torch.cuda.empty_cache()
    del model, tokenizer
    gc.collect()

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()

    return result_dict


def dooneeval(modeltoeval, valid_dataloader, result_dict, i, path, args):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=args.stemmer)
    if isinstance(modeltoeval, torch.nn.parallel.DistributedDataParallel):
        model = modeltoeval.module
    else:
        model = modeltoeval
    model.eval()
    allentnumintar = 0
    allentnuminpre = 0
    hasentnum = 0
    alltar, allpred = [], []
    with torch.no_grad():
        logger.info(len(valid_dataloader))
        for step, batch in tqdm(enumerate(valid_dataloader)):
            #logger.info(step)
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
            sen, target, preds = model._generative_step(inputs)
            sennum = len(sen)
            for ii in range(sennum):
                thissen, thistar, thispred = sen[ii], target[ii], preds[ii]
                if thistar == 'end':
                    continue
                allentintar = thistar.lower().split(',')
                alleninpred = thispred.lower().split(',')

                allentnumintar += len(allentintar)
                allentnuminpre += len(alleninpred)
                for j in range(len(allentintar)):
                    if allentintar[j] in alleninpred:
                        hasentnum += 1
                alltar.append(thistar)
                allpred.append(thispred)
    if allentnuminpre!=0 and allentnumintar!=0:
        p = float(hasentnum) / float(allentnuminpre)
        r = float(hasentnum) / float(allentnumintar)
        if p + r != 0.0:
            f1score = 2 * p * r / (p + r)
        else:
            f1score = 0.0
    else:
        f1score = 0.0
    r1s, r2s, rls = [], [], []
    for j in range(len(alltar)):
        tar = alltar[j]
        pred = allpred[j]
        rouge_score = scorer.score(tar, pred)
        r1s.append(rouge_score["rouge1"].fmeasure)
        r2s.append(rouge_score["rouge2"].fmeasure)
        rls.append(rouge_score["rougeLsum"].fmeasure)
    r1 = np.mean(r1s)
    r2 = np.mean(r2s)
    rl = np.mean(rls)
    logger.info('----Validation Results Summary----')
    logger.info(f1score)
    logger.info(r1)
    logger.info(r2)
    logger.info(rl)

    result_dict['val_F1'].append(f1score)
    result_dict['val_r1'].append(r1)
    result_dict['val_r2'].append(r2)
    result_dict['val_rl'].append(rl)
    if result_dict['val_F1'][-1] > result_dict['best_val_F1']:
        logger.info("{} epoch, best epoch was updated! valid_F1: {: >4.5f}".format(i,result_dict['val_F1'][-1]))
        result_dict["best_val_F1"] = result_dict['val_F1'][-1]
        meanR = (r1 + r2 + rl) / 3
        result_dict["best_val_meanR"] = meanR
        if not os.path.exists(path):
            os.mkdir(path)
        model_to_save = model.module if hasattr(model, 'module') else model
        ckpt = {
            "promptnumber": model_to_save.promptnumber,
            "promptembedding": model_to_save.promptembedding
        }
        torch.save(ckpt, os.path.join(path, "bestckpt_prompt"))
        torch.save(model.state_dict(), os.path.join(path, "bestckpt_full_model"))
