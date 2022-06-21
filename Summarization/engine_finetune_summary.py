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
from models_summarization.model_soft import *
from dataset_finetune_summary import *

logger = logging.getLogger('root')

def train(tokenizer, model, train_dataset, valid_dataset, logger, args):
    # total step
    step_tot = (len(
        train_dataset) // args.gradient_accumulation_steps_summary // args.batch_size_per_gpu_summary // args.n_gpu) * args.max_epoch_summary
    train_sampler = data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(
        train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)

    train_dataloader = get_dataloader(tokenizer, args.num_workers_summary, train_dataset, args.batch_size_per_gpu_summary, args.max_length,
                                      args.max_guidance_length, train_dataset.tokenizer.pad_token_id, train_sampler, args)
    valid_dataloader = get_dataloader(tokenizer, args.num_workers_summary, valid_dataset, args.valid_size_per_gpu_summary, args.max_length,
                                      args.max_guidance_length, valid_dataset.tokenizer.pad_token_id, valid_sampler, args)
    if args.big_testset or args.full_testset:
        test_sampler = SequentialSampler(args.test_dataset)
        test_dataloader = get_dataloader(tokenizer, args.num_workers_summary, args.test_dataset, args.valid_size_per_gpu_summary, args.max_length,
                                      args.max_guidance_length, args.test_dataset.tokenizer.pad_token_id, test_sampler, args)
    
    base_optimizer_arguments = {
        "lr": args.lr_summary,
        "clip_threshold": args.max_grad_norm_summary,
        "decay_rate": -0.8,
        "weight_decay": args.weight_decay_summary,
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

    if args.zero_shot:
        dooneeval(model, valid_dataloader, scaler, result_dict, logger, 0, args)
        return result_dict

    global_step = 0

    if args.big_testset:
        # save the model on the best validation set
        args.save_model = True
    for i in range(args.max_epoch_summary):
        thisevalstep = args.eval_step
        logger.info(i)
        model.train()
        result_dict['epoch'] = i
        allloss = []
        ents = []
        for step, batch in enumerate(train_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device),
                      "ents_ids": batch[4].to(args.device), "ents_mask": batch[5].to(args.device),
                      "predents_ids": batch[4].to(args.device), "predents_mask": batch[5].to(args.device)}
            for k in range(inputs["ents_ids"].shape[0]):
                for l in range(inputs["ents_ids"].shape[1]):
                    ent = inputs["ents_ids"][k,l].item()
                    ents.append(ent)
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

            if step % args.gradient_accumulation_steps_summary == 0 or step == len(train_dataloader) - 1:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.model == 'BartMixPromptUnfreeze':
                        for name, param in model.named_parameters():
                            if "shared" in name:
                                for k in range(param.grad.shape[0]):
                                    if not(k in ents):
                                        param.grad[k,:] = 0
                    optimizer.step()
                    ents = []
                if scheduler != None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [0, -1] and global_step % args.log_step_finetune == 0:
                    logger.info("step: %d, schedule: %.3f, loss: %.6f, " % (
                        global_step, global_step / max(1,step_tot), np.average(allloss)))

                if args.local_rank in [0, -1] and global_step % thisevalstep == 0:
                    print("not eval!!!")
                    model.train()

        logger.info("finish one epoch")
        if args.local_rank in [0, -1]:
            # if i >= 8:
            # do after every epoch
            dooneeval(model, valid_dataloader, scaler, result_dict, logger, i, args)
            model.train()
    # after everything, do it with test:
    if args.big_testset or args.full_testset:
        if args.model == 'T5Finetune':
            path = args.model_save_path + 'full_weights'
            model.load_state_dict(torch.load(path))
            print("loaded the full model weights!", path)
        else:
            path = args.model_save_path + 'bestckpt'
            best_val_ckpt = torch.load(path)
            model.promptnumber = best_val_ckpt["promptnumber"]
            model.promptembedding = nn.parameter.Parameter(best_val_ckpt["promptembedding"])
            print("loaded the model prompt!", path)
        # no need to save again
        args.save_model = False
        # initialize new result_dict to save results
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
        result_dict['epoch'] = args.max_epoch_summary
        dooneeval(model, test_dataloader, scaler, result_dict, logger, args.max_epoch_summary, args)
    torch.cuda.empty_cache()
    del model, optimizer, scheduler, scaler, train_dataloader, valid_dataloader,
    gc.collect()
    
    return result_dict


def get_dataloader(tokenizer, num_workers, dataset, batch_size, max_len, max_guidance_len, pad_id, sampler, args):
    collate_fn = SmartBatchingCollate(
        args = args,
        tokenizer = tokenizer,
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


def dooneeval(modeltoeval, valid_dataloader, scaler, result_dict, logger, i, args):
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
            # logger.info(step)
            if step % args.log_step_finetune == 0:
                logger.info("step: %d, schedule: %.3f" % (step, step / len(valid_dataloader)))
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device),
                      "ents_ids": batch[4].to(args.device), "ents_mask": batch[5].to(args.device),
                      "predents_ids": batch[6].to(args.device), "predents_mask": batch[7].to(args.device)}
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
            if not os.path.exists(args.model_save_path):
                os.mkdir(args.model_save_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            if args.model == 'T5Finetune':
                path = args.model_save_path + 'full_weights'
                torch.save(model_to_save.state_dict(), path)
                print("saved the full model weights!", path)
            else:
                path = args.model_save_path + 'bestckpt'
                ckpt = {
                    "promptnumber": model_to_save.promptnumber,
                    "promptembedding": model_to_save.promptembedding
                }
                torch.save(ckpt, path)
                print("saved the model prompt!", path)
    
    return result_dict


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
