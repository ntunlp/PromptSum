import gc
import logging
import os
import numpy as np
import torch.nn.functional as F
import sys
import datasets
sys.path.append("./T5PromptNER/")
import spacy
import nltk
import pickle

from transformers import (AdamW, BertConfig, BertForTokenClassification, BertTokenizer, get_linear_schedule_with_warmup)
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from torch.utils import data
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from rouge_score import rouge_scorer
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.grad_scaler import ShardedGradScaler
from torch.cuda.amp import autocast as autocast
from nltk.tokenize import sent_tokenize

from dataset_pretrain import *
from model_pretrain import *


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



def pretrain_model(dataset_args, args):
    print("Pre-training entity tagger...")

    ### train
    gradient_accumulation_steps = args.gradient_accumulation_steps_pretrain
    train_batch_size = args.batch_size_per_gpu_pretrain
    eval_batch_size = args.valid_size_per_gpu_pretrain
    num_train_epochs = args.max_epoch_pretrain ### epochs for training tagger
    learning_rate = args.lr_pretrain
    if args.pretrain_all_weights:
        print("pretrain_all_weights")
        learning_rate = 5e-5
    weight_decay = args.weight_decay_pretrain
    max_seq_length = args.max_length
    num_workers = args.num_workers_pretrain
    max_grad_norm = args.max_grad_norm_pretrain
    log_step = args.log_step_pretrain
    eval_step = args.eval_step
    model_name = args.model_name

    t5model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir = args.cache_path)
    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir = args.cache_path)
    model = T5forPretrain(args, t5model, tokenizer)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("The model has {} trainable parameters".format(n_params))

    ##### load from conll ckpt or simply initializing?
    ifuseconll = False
    if ifuseconll:
        allckpt = torch.load("./T5PromptNER/bestckpt")
        model.promptnumber = allckpt["promptnumber"]
        model.promptembedding = allckpt["promptembedding"]
        model.promptnumberforsum = allckpt["promptnumber"]
        model.promptembeddingforsum = allckpt["promptembedding"]
    else:
        promptnumber = 300
        taskname = "name entity recognition"
        promptembedding = getpromptembedding(model, tokenizer, promptnumber, taskname)
        print("prompt", promptembedding.shape)
        model.set_prompt_embedding(promptnumber, promptembedding)

        tasknamesum = "summarization"
        promptembeddingforsum = getpromptembedding(model, tokenizer, promptnumber, tasknamesum)
        model.set_prompt_embedding_sum(promptnumber, promptembeddingforsum)

    model.to(args.device)

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    spacy_nlp = spacy.load("en_core_web_sm")

    # data
    full_data = datasets.load_dataset(*dataset_args, cache_dir=args.dataset_cache_dir)
    train_data = full_data['train']
    valid_data = full_data['validation']
    train_texts = [x[args.text_key] for x in train_data]
    val_texts = [x[args.text_key] for x in valid_data]
    p = np.random.permutation(len(val_texts))
    val_texts = [val_texts[x] for x in p]
    val_texts = val_texts[:1000]
    print(len(train_texts), len(val_texts))
    if args.debug_pretrain:
        train_texts = train_texts[:10]
        val_texts = val_texts[:10]
        print(len(train_texts), len(val_texts))

    # build data
    if args.build_salient_entities:
        train_texts, train_target, train_ents = find_salient_sentences_and_entities(train_texts, scorer, spacy_nlp, args)
        train_data = train_texts, train_target, train_ents
        train_path = "t5_tagger_pretraining_data/{}_train_{}.pkl".format(dataset_args[0], len(train_texts))
        with open(train_path, "wb") as f:
            pickle.dump(train_data, f)
            print("saved the pre-training train data")
        val_texts, valid_target, val_ents = find_salient_sentences_and_entities(val_texts, scorer, spacy_nlp, args)
        val_data = val_texts, valid_target, val_ents
        val_path = "t5_tagger_pretraining_data/{}_val_{}.pkl".format(dataset_args[0], len(val_texts))
        with open(val_path, "wb") as f:
            pickle.dump(val_data, f)
            print("saved the pre-training val data")
        raise Exception
    else:
        train_path = "t5_tagger_pretraining_data/{}_train_{}.pkl".format(dataset_args[0], args.pretraining_train_size)
        with open(train_path, "rb") as f:
            train_data = pickle.load(f)
        print("load the pre-training train data")
        train_texts, train_target, train_ents = train_data
        print(len(train_texts))
        val_path = "t5_tagger_pretraining_data/{}_val_{}.pkl".format(dataset_args[0], args.pretraining_val_size)
        with open(val_path, "rb") as f:
            val_data = pickle.load(f)
        print("load the pre-training val data")
        val_texts, valid_target, val_ents = val_data
        print(len(val_texts))
        if args.debug_pretrain:
            train_texts = train_texts[:10]
            train_target = train_target[:10]
            train_ents = train_ents[:10]
            val_texts = val_texts[:10]
            valid_target = valid_target[:10]
            val_ents = val_ents[:10]
            print(len(train_texts), len(val_texts))

    # datasets
    train_dataset = T5DatasetPretrain(train_texts, train_ents, train_target, max_seq_length, tokenizer, args)
    valid_dataset = T5DatasetPretrain(val_texts, val_ents, valid_target, max_seq_length, tokenizer, args)

    if args.local_rank != -1:
        torch.distributed.barrier()

    # samplers
    train_sampler = data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)

    # loaders
    train_dataloader = get_dataloader_tag_pretrain(num_workers, train_dataset, train_batch_size, max_seq_length, train_dataset.tokenizer.pad_token_id, train_sampler)
    valid_dataloader = get_dataloader_tag_pretrain(num_workers, valid_dataset, eval_batch_size, max_seq_length, valid_dataset.tokenizer.pad_token_id, valid_sampler)

    logger.info("Begin pre-train...")

    base_optimizer_arguments = {
        "lr": learning_rate,
        "clip_threshold": max_grad_norm,
        "decay_rate": -0.8,
        "weight_decay": weight_decay,
        "scale_parameter": False,
        "relative_step": False
    }
    optimizer = Adafactor
    optimizer = OSS(params=filter(lambda p: p.requires_grad, model.parameters()), optim=optimizer, **base_optimizer_arguments)
    model = ShardedDDP(model, optimizer)
    #optimizer = Adafactor(params=filter(lambda p: p.requires_grad, model.parameters()), **base_optimizer_arguments)

    #scaler = ShardedGradScaler()
    scaler = None
    scheduler = None

    Best_F1 = -1
    Best_val_meanR = -100.0
    result_dict = {
        'epoch': [],
        'val_F1': [],
        #'best_val_F1': Best_F1,
        'val_r1': [],
        'val_r2': [],
        'val_rl': [],
        'val_r1_sum': [],
        'val_r2_sum': [],
        'val_rl_sum': [],
        'val_meanR': [],
        'best_val_meanR': Best_val_meanR
    }
    global_step = 0
    output_dir = "t5_tagger_pretrained_ckpt/"
    print("\nEpoch 0 validation:")
    dooneevalforpretrain(model, valid_dataloader, scaler, result_dict, 0, output_dir, args)
    lossentcoff = 1.0
    losssumcoff = 1.0
    for i in range(num_train_epochs):
        logger.info(i)
        model.train()
        result_dict['epoch'] = i
        alllossent = []
        alllosssum = []
        for step, batch in enumerate(train_dataloader):
            #print(step)
            inputs_all = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                          "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device),
                          "entity_ids": batch[4].to(args.device), "entity_mask": batch[5].to(args.device)}
            if scaler is not None:
                with autocast():
                    lossent, losssum = model(inputs_all)
            else:
                lossent, losssum = model(inputs_all)

            loss = lossent * lossentcoff + losssum * losssumcoff
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            alllossent.append(lossent.item())
            alllosssum.append(losssum.item())
            del inputs_all
            gc.collect()

            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler != None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [0, -1] and global_step % log_step == 0:
                    logger.info("step: %d,  lossent: %.6f, losssum: %.6f" % (global_step,  np.average(alllossent), np.average(alllosssum)))
                    allloss = []

                if args.local_rank in [0, -1] and global_step % eval_step == 0:
                    dooneevalforpretrain(model, valid_dataloader, scaler, result_dict, i, output_dir, args)
                    model.train()

        logger.info("finish one epoch")
        if args.local_rank in [0, -1]:
            dooneevalforpretrain(model, valid_dataloader, scaler, result_dict, i, output_dir, args)
            model.train()

    torch.cuda.empty_cache()
    del model, tokenizer
    gc.collect()

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()


def find_salient_sentences_and_entities(texts, scorer, spacy_nlp, args):
    print("finding the salient entities...")
    n = 3
    # index = 0
    all_texts, all_top_sen, all_ents = [], [], []
    for i in tqdm(range(len(texts))):
        # if index > 100:
        #     break
        # index += 1
        text = texts[i]
        sents = nltk.sent_tokenize(text)
        sents = sents[:40]
        r1s = []
        for j in range(len(sents)):
            sent = sents[j]
            rest = " ".join(sents[:j] + sents[(j+1):])
            rouge_scores = scorer.score(rest, sent)
            r1 = rouge_scores["rouge1"].fmeasure
            r1s.append(r1)
        idx = np.argsort(np.array(r1s))[::-1]
        # top sents
        top_idx = idx[:n]
        top_idx.sort()
        top_sents = [sents[i] for i in top_idx]
        top_sents = " ".join(top_sents)
        all_top_sen.append(top_sents)
        # top entities
        ents = spacy_nlp(top_sents).ents
        allents = [ent.text for ent in ents]
        if allents == []:
            allents = ["none"]
        all_ents.append(allents)
        # text
        bottom_idx = idx[n:]
        bottom_idx.sort()
        rest_sents = [sents[i] for i in bottom_idx]
        rest = " ".join(rest_sents)
        all_texts.append(rest)

    return all_texts, all_top_sen, all_ents


def dooneevalforpretrain(modeltoeval, valid_dataloader, scaler, result_dict, i, path, args):
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
    alltarsum, allpredsum = [], []
    with torch.no_grad():
        logger.info(len(valid_dataloader))
        for step, batch in tqdm(enumerate(valid_dataloader)):
            inputs_all = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                          "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device),
                          "entity_ids": batch[4].to(args.device), "entity_mask": batch[5].to(args.device)}
            if scaler is not None:
                with autocast():
                    sensum, targetsum, predssum, sen, target, preds = model._generative_step(inputs_all)
            else:
                sensum, targetsum, predssum, sen, target, preds = model._generative_step(inputs_all)
            del inputs_all
            gc.collect()
            alltarsum.extend(targetsum)
            allpredsum.extend(predssum)
            ###### how to evaluate both sum and ent?
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

    r1sforsum, r2sforsum, rlsforsum = [], [], []
    for j in range(len(alltarsum)):
        label = alltarsum[j]
        summary = allpredsum[j]
        if args.highlights:
            label = "\n".join(sent_tokenize(label))
            summary = "\n".join(sent_tokenize(summary))
        rouge_score = scorer.score(label, summary)
        r1sforsum.append(rouge_score["rouge1"].fmeasure)
        r2sforsum.append(rouge_score["rouge2"].fmeasure)
        rlsforsum.append(rouge_score["rougeLsum"].fmeasure)
    r1forsum = np.mean(r1sforsum)
    r2forsum = np.mean(r2sforsum)
    rlforsum = np.mean(rlsforsum)

    logger.info('----Validation Results Summary----')
    message = "\nENTITY eval: F1: {:.4f}, R-1: {:.4f}, R-2: {:.4f}, R-L: {:.4f} " \
              "\nSUMMARY eval: R-1: {:.4f}, R-2: {:.4f}, R-L: {:.4f}".format(
        f1score, r1, r2, rl, r1forsum, r2forsum, rlforsum
    )
    logger.info(message)

    result_dict['val_F1'].append(f1score)
    result_dict['val_r1'].append(r1)
    result_dict['val_r2'].append(r2)
    result_dict['val_rl'].append(rl)
    result_dict['val_r1_sum'].append(r1forsum)
    result_dict['val_r2_sum'].append(r2forsum)
    result_dict['val_rl_sum'].append(rlforsum)
    cur_val_meanR = (r1 + r2 + rl + r1forsum + r2forsum + rlforsum) / 6
    result_dict['val_meanR'].append(cur_val_meanR)

    if result_dict['val_meanR'][-1] > result_dict['best_val_meanR']:
        logger.info("{} epoch, best epoch was updated! valid_meanR of sum and ent: {: >4.5f}".format(i,result_dict['val_meanR'][-1]))
        result_dict["best_val_meanR"] = result_dict['val_meanR'][-1]
        #result_dict["best_val_F1"] = result_dict['val_F1'][-1]
        if not os.path.exists(path):
            os.mkdir(path)
        model_to_save = model.module if hasattr(model, 'module') else model
        promptembedding = model_to_save.promptembedding.detach().cpu()
        promptembeddingforsum = model_to_save.promptembeddingforsum.detach().cpu()
        ckpt = {
            "promptnumber": model_to_save.promptnumber,
            "promptembedding": promptembedding,
            "promptnumberforsum": model_to_save.promptnumberforsum,
            "promptembeddingforsum":promptembeddingforsum,
        }
        torch.save(ckpt, os.path.join(path, "bestckpt_prompt"))
        torch.save(model.state_dict(), os.path.join(path, "bestckpt_full_model"))
