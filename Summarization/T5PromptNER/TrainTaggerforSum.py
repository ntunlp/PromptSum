import gc
import logging
import os
import numpy as np
import torch.nn.functional as F
from transformers import (AdamW, BertConfig, BertForTokenClassification, BertTokenizer,
                                  get_linear_schedule_with_warmup)
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from torch.utils import data
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from rouge_score import rouge_scorer

import sys
import datasets
sys.path.append("./T5PromptNER/")
from NERDataset import *
from NERModel import *
import spacy
import nltk
import pickle

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def getfilewithlabel(file, filewithfakelabel):

    fin = open(file,'r')
    alldata = []
    while True:
        oneline = fin.readline().strip()
        if not oneline:
            break
        alldata.append(oneline)
    fin.close()

    #print(len(alldata))

    fo = open(filewithfakelabel, 'w')

    for onedata in alldata:
        fo.write(onedata+"\tend\n")
    fo.close()
    return alldata


def getdocandent(docfile,sum_y_pred):

    f = open(docfile,'r')
    alldoc = []
    while True:
        oneline = f.readline().strip()
        if not oneline:
            break
        alldoc.append(oneline)
    f.close()
    resfortrain = []
    resforvalid = []
    trainsize = len(alldoc) // 2
    allres = []
    for i in range(len(alldoc)):
        if i < trainsize:
            resfortrain.append(alldoc[i] + "\t" + sum_y_pred[i])
        else:
            resforvalid.append(alldoc[i] + "\t" + sum_y_pred[i])
        allres.append(alldoc[i] + "\t" + sum_y_pred[i])
    return allres, resfortrain, resforvalid

def get_predict_label_for_sum(args, doc_sum_path, sumpath, spacy_nlp):

    #####handle sumfile to fake conll format and use NER model to label it
    allpreds = []
    if not args.if_spacy:
        sumwithfakelabel = doc_sum_path + "sumwithfakelabel.txt"
        allsumwithfakelabeldata = getfilewithlabel(sumpath, sumwithfakelabel)
        model_name = "google/t5-v1_1-large"
        t5model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir="/data/qin/hf_models/t5-v1-large/")
        tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir="/data/qin/hf_models/t5-v1-large/")
        model = T5forNER(args, t5model, tokenizer)
        test_dataset = T5NERDatasetConll(sumwithfakelabel, 512, tokenizer)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = get_dataloader_tag(4, test_dataset, 8, 512, test_dataset.tokenizer.pad_token_id, test_sampler)

        allckpt = torch.load("./T5PromptNER/bestckpt")
        model.promptnumber = allckpt["promptnumber"]
        model.promptembedding = allckpt["promptembedding"]
        #print(model.promptnumber)
        #print(model.promptembedding.shape)

        model.to(args.device)
        model.eval()

        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):
                inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                          "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
                sen, target, preds = model._generative_step(inputs)
                allpreds.extend(preds)

        torch.cuda.empty_cache()
        del model, tokenizer, test_dataloader
        gc.collect()

        assert len(allpreds) == len(allsumwithfakelabeldata)
    else:
        allpreds = [] ##should be separated by ','
        fin = open(sumpath, 'r')
        index = 0
        while True:
            oneline = fin.readline().strip()
            if not oneline:
                break
            ents = spacy_nlp(oneline).ents
            allents = [ent.text for ent in ents]
            if allents == []:
                allents = ["none"]

            input_guidance = ','.join(list(dict.fromkeys(allents)))     ##### unique
            #input_guidance = ','.join(allents)  ##### not unique
            allpreds.append(input_guidance)
            index += 1
        fin.close()
    return allpreds

def get_doc_label(sum_y_pred, docfile):

    alldocres, resfortrain, resforvalid = getdocandent(docfile, sum_y_pred)

    allentityfortrain = []
    for i in range(len(resfortrain)):
        onedata = resfortrain[i].split('\t')
        onedoc = onedata[0]
        oneent = onedata[1]
        allentityfortrain.append([onedoc, oneent])

    allentityforvalid = []
    for i in range(len(resforvalid)):
        onedata = resforvalid[i].split('\t')
        onedoc = onedata[0]
        oneent = onedata[1]
        allentityforvalid.append([onedoc, oneent])


    alldocandlabel = []
    for i in range(len(alldocres)):
        onedata = alldocres[i].split('\t')
        onedoc = onedata[0]
        oneent = onedata[1]
        alldocandlabel.append([onedoc, oneent])

    return alldocandlabel,allentityfortrain,allentityforvalid

def get_train_valid(alldocandlabel, doc_sum_path, allentityfortrain, allentityforvalid):

    docwithlabel_train = doc_sum_path + "docwithlabel_train.txt"
    docwithlabel_vaid = doc_sum_path + "docwithlabel_valid.txt"

    fout = open(docwithlabel_train, 'w')
    fout_1 = open(docwithlabel_vaid, 'w')

    halfsize = len(alldocandlabel) // 2
    # print(halfsize)
    for aa in range(len(alldocandlabel)):
        onedata = alldocandlabel[aa]
        # if aa % 2 == 0:
        #     fout.write(onedata[0] + "\t" + onedata[1] + "\n")
        # else:
        #     fout_1.write(onedata[0] + "\t" + onedata[1] + "\n")
        if aa < halfsize:
            fout.write(onedata[0] + "\t" + onedata[1] + "\n")
        else:
            fout_1.write(onedata[0] + "\t" + onedata[1] + "\n")
    fout.close()
    fout_1.close()

    ####save train ent
    train_ent = doc_sum_path + "trainent.txt"
    fe = open(train_ent, 'w')
    for i in range(len(allentityfortrain)):
        if allentityfortrain[i][1] != []:
            fe.write(allentityfortrain[i][0] + "\t" + allentityfortrain[i][1] + '\n')
        else:
            fe.write(allentityfortrain[i][0] + "\tnone\n")
    fe.close()

    valid_ent = doc_sum_path + "valident.txt"
    fe = open(valid_ent, 'w')
    for i in range(len(allentityforvalid)):
        if allentityforvalid[i][1] != []:
            fe.write(allentityforvalid[i][0] + "\t" + allentityforvalid[i][1] + '\n')
        else:
            fe.write(allentityforvalid[i][0] + "\tnone\n")
    fe.close()

    return docwithlabel_train, docwithlabel_vaid


def dooneeval(modeltoeval,valid_dataloader,args,result_dict,i,path):
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


def finetune_model(trainfile, validfile, args):
    print("Fine-tuning entity tagger...")

    print(trainfile, validfile)

    ###train
    gradient_accumulation_steps = 2
    train_batch_size = 2
    eval_batch_size = 4
    num_train_epochs = 60 ### epochs for training tagger
    learning_rate = 5e-1
    weight_decay = 1e-5
    max_seq_length = 512
    num_workers = 4
    max_grad_norm = 1.0
    log_step = 1
    model_name = "google/t5-v1_1-large"

    t5model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir="/data/qin/hf_models/t5-v1-large/")
    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir="/data/qin/hf_models/t5-v1-large/")
    model = T5forNER(args, t5model, tokenizer)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("The model has {} trainable parameters".format(n_params))

    ##### load from conll ckpt, from pre-training ckpt, or simply initializing?
    if args.use_pretrain_ckpt:
        print("Loading the pre-trained NER model!")
        
        # full model
        ckpt = torch.load("t5_tagger_pretrained_ckpt/bestckpt_full_model_39k")
        dic = {}
        for x in ckpt.keys():
            if not(x in ["promptnumber", "promptembedding"]):
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

    train_dataset = T5NERDatasetConll(trainfile, max_seq_length, tokenizer)
    valid_dataset = T5NERDatasetConll(validfile, max_seq_length, tokenizer)

    if args.local_rank != -1:
        torch.distributed.barrier()

    train_sampler = data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)

    train_dataloader = get_dataloader_tag(num_workers, train_dataset, train_batch_size, max_seq_length, train_dataset.tokenizer.pad_token_id, train_sampler)
    valid_dataloader = get_dataloader_tag(num_workers, valid_dataset, eval_batch_size, max_seq_length, valid_dataset.tokenizer.pad_token_id, valid_sampler)

    #####the path of tuned model
    pos = trainfile.find("docwithlabel_train")
    foldername = trainfile[0:pos]
    print(foldername)
    #raise Exception
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
                    logger.info("step: %d,  loss: %.6f" % (global_step,  np.average(allloss)))

                if args.local_rank in [0, -1] and global_step % thisevalstep == 0:
                    print("only eval after every epoch")
                    model.train()

        logger.info("finish one epoch")
        if args.local_rank in [0, -1]:
            dooneeval(model, valid_dataloader, args, result_dict, i, output_dir)

    torch.cuda.empty_cache()
    del model, tokenizer
    gc.collect()

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()

    return result_dict


def pretrain_model(dataset_args, args):
    print("Pre-training entity tagger...")

    ###train
    gradient_accumulation_steps = 4
    train_batch_size = 1
    eval_batch_size = 4
    num_train_epochs = 5 ### epochs for training tagger
    learning_rate = 5e-1
    if args.pretrain_all_weights:
        learning_rate = 5e-5
    weight_decay = 0
    max_seq_length = 512
    num_workers = 4
    max_grad_norm = 1.0
    log_step = 50
    eval_step = 1000
    model_name = "google/t5-v1_1-large"

    t5model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir="/data/qin/hf_models/t5-v1-large/")
    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir="/data/qin/hf_models/t5-v1-large/")
    model = T5forNER(args, t5model, tokenizer)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("The model has {} trainable parameters".format(n_params))

    ##### load from conll ckpt or simply initializing?
    ifuseconll = False
    if ifuseconll:
        allckpt = torch.load("./T5PromptNER/bestckpt")
        model.promptnumber = allckpt["promptnumber"]
        model.promptembedding = allckpt["promptembedding"]
    else:
        promptnumber = 300
        taskname = "name entity recognition"
        promptembedding = getpromptembedding(model, tokenizer, promptnumber, taskname)
        print("prompt", promptembedding.shape)
        model.set_prompt_embedding(promptnumber, promptembedding)

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
    #train_texts = train_texts[:100]
    #val_texts = val_texts[:100]

    # build data
    if args.build_salient_entities:
        train_texts, train_ents = find_salient_sentences_and_entities(train_texts, scorer, spacy_nlp, args)
        train_data = train_texts, train_ents
        train_path = "t5_tagger_pretraining_data/{}_train_{}.pkl".format(dataset_args[0], len(train_texts))
        with open(train_path, "wb") as f:
            pickle.dump(train_data, f)
            print("saved the pre-training train data")
        val_texts, val_ents = find_salient_sentences_and_entities(val_texts, scorer, spacy_nlp, args)
        val_data = val_texts, val_ents
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
        train_texts, train_ents = train_data
        print(len(train_texts))
        val_path = "t5_tagger_pretraining_data/{}_val_{}.pkl".format(dataset_args[0], args.pretraining_val_size)
        with open(val_path, "rb") as f:
            val_data = pickle.load(f)
        print("load the pre-training val data")
        val_texts, val_ents = val_data
        print(len(val_texts))

    # datasets
    train_dataset = T5NERDataset(train_texts, train_ents, max_seq_length, tokenizer, args)
    valid_dataset = T5NERDataset(val_texts, val_ents, max_seq_length, tokenizer, args)

    if args.local_rank != -1:
        torch.distributed.barrier()

    # samplers
    train_sampler = data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)

    # loaders
    train_dataloader = get_dataloader_tag(num_workers, train_dataset, train_batch_size, max_seq_length, train_dataset.tokenizer.pad_token_id, train_sampler)
    valid_dataloader = get_dataloader_tag(num_workers, valid_dataset, eval_batch_size, max_seq_length, valid_dataset.tokenizer.pad_token_id, valid_sampler)

    logger.info("Begin pre-train...")

    base_optimizer_arguments = {
        "lr": learning_rate,
        "clip_threshold": max_grad_norm,
        "decay_rate": -0.8,
        "weight_decay": weight_decay,
        "scale_parameter": False,
        "relative_step": False
    }
    optimizer = Adafactor(params=filter(lambda p: p.requires_grad, model.parameters()), **base_optimizer_arguments)

    Best_F1 = -1
    Best_val_meanR = 0.0
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
    output_dir = "t5_tagger_pretrained_ckpt/"
    print("\nEpoch 0 validation:")
    dooneeval(model, valid_dataloader, args, result_dict, 0, output_dir)
    for i in range(num_train_epochs):
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
                    logger.info("step: %d,  loss: %.6f" % (global_step,  np.average(allloss)))
                    allloss = []

                if args.local_rank in [0, -1] and global_step % eval_step == 0:
                    dooneeval(model, valid_dataloader, args, result_dict, i, output_dir)
                    model.train()

        logger.info("finish one epoch")
        if args.local_rank in [0, -1]:
            dooneeval(model, valid_dataloader, args, result_dict, i, output_dir)
            model.train()

    torch.cuda.empty_cache()
    del model, tokenizer
    gc.collect()

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()


def find_salient_sentences_and_entities(texts, scorer, spacy_nlp, args):
    print("finding the salient entities...")
    n = 3
    all_texts, all_ents = [], []
    for i in tqdm(range(len(texts))):
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

    return all_texts, all_ents
