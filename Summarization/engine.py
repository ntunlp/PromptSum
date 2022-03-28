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

from model import *
from dataset import *
from utils import *



def train(args, model, train_dataset, valid_dataset, logger):
    # total step
    step_tot = (len(
        train_dataset) // args.gradient_accumulation_steps // args.batch_size_per_gpu // args.n_gpu) * args.max_epoch
    warmup_steps_total = step_tot * args.warmup_steps
    train_sampler = data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(
        train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)

    train_dataloader = get_dataloader(args, args.num_workers, train_dataset, args.batch_size_per_gpu, args.max_length,
                                      args.max_guidance_length, train_dataset.tokenizer.pad_token_id, train_sampler)
    valid_dataloader = get_dataloader(args, args.num_workers, valid_dataset, args.valid_size_per_gpu, args.max_length,
                                      args.max_guidance_length, valid_dataset.tokenizer.pad_token_id, valid_sampler)

    base_optimizer_arguments = {
        "lr": args.lr, 
        "clip_threshold": args.max_grad_norm, 
        "decay_rate": -0.8,
        "weight_decay": args.weight_decay,
        "scale_parameter": False, 
        "relative_step": False
    }
    optimizer = Adafactor
    if args.n_gpu > 1: # distributed training
        optimizer = OSS(params=filter(lambda p: p.requires_grad, model.parameters()), optim=optimizer,
                        **base_optimizer_arguments)
        # distributed training
        model = ShardedDDP(model, optimizer)
    else:
        optimizer = optimizer(params=filter(lambda p: p.requires_grad, model.parameters()), **base_optimizer_arguments)
    model.train()
    #scaler = ShardedGradScaler()
    scheduler = None
    scaler = None

    logger.info("Begin train...")
    logger.info("We will train model in %d steps" % step_tot)

    result_dict = {
        'epoch': [],
        'val_mean_rouge': [],
        "best_val_mean_rouge": 0.0,
        "val_rouge1": 0.0,
        "val_rouge2": 0.0,
        "val_rougeL": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }
    global_step = 0
    for i in range(args.max_epoch):
        thisevalstep = args.eval_step
        logger.info(i)
        model.train()
        result_dict['epoch'] = i
        allloss = []
        for step, batch in enumerate(train_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device),
                      "input_ents": batch[4].to(args.device), "ents_mask": batch[5].to(args.device)}
            if scaler is not None:
                with autocast():
                    loss = model(inputs)
            else:
                loss  = model(inputs)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            allloss.append(loss.item())

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler != None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [0, -1] and global_step % args.log_step == 0:
                    logger.info("step: %d, schedule: %.3f, loss: %.6f, " % (
                        global_step, global_step / step_tot, np.average(allloss)))

                if args.local_rank in [0, -1] and global_step % thisevalstep == 0:
                    print("not eval!!!")
                    model.train()

        logger.info("finish one epoch")
        if args.local_rank in [0, -1]:
            # if i >= 8:
            # do after every epoch
            dooneeval(args, model, valid_dataloader, scaler, result_dict, logger,i)
            model.train()

        if args.train_sample:
            logger.info("sampling...")
            logger.info("sampled")

    torch.cuda.empty_cache()
    del model, optimizer, scheduler, scaler, train_dataloader, valid_dataloader,
    gc.collect()
    
    return result_dict


def get_dataloader(args, num_workers,dataset, batch_size, max_len, max_guidance_len, pad_id, sampler):
    collate_fn = SmartBatchingCollate(
        args = args,
        max_length=max_len,
        max_guidance_length=max_guidance_len,
        pad_token_id=pad_id
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def dooneeval(args, modeltoeval, valid_dataloader, scaler, result_dict, logger, i):
    if isinstance(modeltoeval, torch.nn.parallel.DistributedDataParallel):
        model = modeltoeval.module
    else:
        model = modeltoeval
    model.eval()
    logger.info("Do one eval!")
    allytrue = []
    allypred = []
    with torch.no_grad():
        logger.info(len(valid_dataloader))
        for step, batch in enumerate(valid_dataloader):
            logger.info(step)
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device),
                      "input_ents": batch[4].to(args.device), "ents_mask": batch[5].to(args.device)}
            if scaler is not None:
                with autocast():
                    sen, target, preds = model._generative_step(inputs)
                    tarres, predres = target, preds
                    allytrue.extend(tarres)
                    allypred.extend(predres)
            else:
                sen, target, preds = model._generative_step(inputs)
                tarres, predres = target, preds
                allytrue.extend(tarres)
                allypred.extend(predres)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer = args.stemmer)
    r1s, r2s, rls = [], [], []
    for j in range(len(allytrue)):
        label = allytrue[j]
        summary = allypred[j]
        if args.highlights:
            label = "\n".join(sent_tokenize(label))
            summary = "\n".join(sent_tokenize(summary))
        rouge_score = scorer.score(label, summary)
        r1s.append(rouge_score["rouge1"].fmeasure)
        r2s.append(rouge_score["rouge2"].fmeasure)
        rls.append(rouge_score["rougeLsum"].fmeasure)
    rouge_score = {
        "rouge1": 100 * np.mean(r1s),
        "rouge2": 100 * np.mean(r2s),
        "rougeLsum": 100 * np.mean(rls)
    }
    logger.info('----Validation Results Summary----')
    logger.info(len(allypred))
    logger.info(rouge_score)
    p, r, f1 = entity_eval(allytrue, allypred)

    # change accordingly
    mean_rouge = (rouge_score["rouge1"] + rouge_score["rouge2"] + rouge_score["rougeLsum"]) / 3
    result_dict['val_mean_rouge'].append(mean_rouge)
    if result_dict['val_mean_rouge'][-1] > result_dict['best_val_mean_rouge']:
        logger.info("{} epoch, best epoch was updated! val_mean_rouge: {: >4.5f}".format(i, result_dict['val_mean_rouge'][-1]))
        result_dict["best_val_mean_rouge"] = result_dict['val_mean_rouge'][-1]
        # also append other rouge scores
        result_dict['val_rouge1'] = rouge_score["rouge1"]
        result_dict['val_rouge2'] = rouge_score["rouge2"]
        result_dict['val_rougeL'] = rouge_score["rougeLsum"]
        
        result_dict['precision'] = p
        result_dict['recall'] = r
        result_dict['f1'] = f1

        if args.save_model:
            model_to_save = model.module if hasattr(model, 'module') else model
            ckpt = {
                "promptnumber": model_to_save.promptnumber,
                "promptembedding": model_to_save.promptembedding
            }
            torch.save(ckpt, args.save_model_path)
    
    return result_dict


def test(args, test_dataset, logger):

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = get_dataloader(args, args.num_workers, test_dataset, args.test_size_per_gpu, args.max_length,
                                    args.max_guidance_length, test_dataset.tokenizer.pad_token_id,test_sampler)

    t5model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
    model = T5forSummarization(args, t5model, test_dataset.tokenizer)
    allckpt = torch.load(args.save_model_path)
    model.promptnumber = allckpt["promptnumber"]
    model.promptembedding = allckpt["promptembedding"]
    logger.info("load finished!")

    model.to(args.device)
    model.eval()
    #scaler = ShardedGradScaler()
    scaler = None
    allytrue = []
    allypred = []

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
            if scaler is not None:
                with autocast():
                    sen,target,preds = model._generative_step(inputs)
                    tarres, predres = target, preds
                    allytrue.extend(tarres)
                    allypred.extend(predres)
            else:
                sen, target, preds = model._generative_step(inputs)
                tarres, predres = target, preds
                allytrue.extend(tarres)
                allypred.extend(predres)
    rouge = load_metric('rouge')
    rouge_score = rouge.compute(references=allytrue, predictions=allypred)
    logger.info('----Test Results Summary----')
    logger.info(len(allypred))
    logger.info(rouge_score)
    logger.info("test_rouge1: %f", rouge_score["rouge1"].mid.fmeasure)
    logger.info("test_rouge2: %f", rouge_score["rouge2"].mid.fmeasure)
    logger.info("test_rougeL: %f", rouge_score["rougeL"].mid.fmeasure)


def entity_eval(ytrue, ypred):
    spacy_nlp = spacy.load("en_core_web_sm")
    all_p = []
    all_r = []
    all_f1 = []
    for i in tqdm(range(len(ytrue))):
        ents_true = spacy_nlp(ytrue[i]).ents
        ents_true = [ent.text for ent in ents_true]
        ents_pred = spacy_nlp(ypred[i]).ents
        ents_pred = [ent.text for ent in ents_pred]
        p = 0
        r = 0
        f1 = 0
        if len(ents_pred) > 0:
            p = 100 * len([x for x in ents_pred if x in ents_true]) / len(ents_pred)
        else:
            if len(ents_true) == 0:
                p = 100
        if len(ents_true) > 0:
            r = 100 * len([x for x in ents_true if x in ents_pred]) / len(ents_true)
        else:
            if len(ents_pred) == 0:
                r = 100
        if (p + r) > 0:
            f1 = (2 * p * r) / (p + r)
        all_p.append(p)
        all_r.append(r)
        all_f1.append(f1)
    p = np.mean(all_p)
    r = np.mean(all_r)
    f1 = np.mean(all_f1)
    print("\nEntity-level eval, mean precision: {:.4f}, recall: {:.4f}, F-1: {:.4f}".format(p, r, f1))
    
    return p, r, f1
