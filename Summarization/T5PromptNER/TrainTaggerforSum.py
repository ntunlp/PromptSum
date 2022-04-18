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

import sys
sys.path.append("./T5PromptNER/")
from NERDataset import *
from NERModel import *

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
        t5model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir="/data/qin/cache/")
        tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir="/data/qin/cache/")
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
    if isinstance(modeltoeval, torch.nn.parallel.DistributedDataParallel):
        model = modeltoeval.module
    else:
        model = modeltoeval
    model.eval()
    allentnumintar = 0
    allentnuminpre = 0
    hasentnum = 0
    with torch.no_grad():
        logger.info(len(valid_dataloader))
        for step, batch in enumerate(valid_dataloader):
            logger.info(step)
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
    if allentnuminpre!=0 and allentnumintar!=0:
        p = float(hasentnum) / float(allentnuminpre)
        r = float(hasentnum) / float(allentnumintar)
        if p + r != 0.0:
            f1score = 2 * p * r / (p + r)
        else:
            f1score = 0.0
    else:
        f1score = 0.0
    logger.info('----Validation Results Summary----')
    logger.info(f1score)

    result_dict['val_F1'].append(f1score)
    if result_dict['val_F1'][-1] > result_dict['best_val_F1']:
        logger.info("{} epoch, best epoch was updated! valid_F1: {: >4.5f}".format(i,result_dict['val_F1'][-1]))
        result_dict["best_val_F1"] = result_dict['val_F1'][-1]
        if not os.path.exists(path):
            os.mkdir(path)
        model_to_save = model.module if hasattr(model, 'module') else model
        ckpt = {
            "promptnumber": model_to_save.promptnumber,
            "promptembedding": model_to_save.promptembedding
        }
        torch.save(ckpt, os.path.join(path, "bestckpt"))

def finetune_model(trainfile, validfile, args):

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

    t5model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir="/data/qin/cache/")
    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir="/data/qin/cache/")
    model = T5forNER(args, t5model, tokenizer)

    ##### load from conll ckpt or simply initializing?
    ifuseconll = True
    if ifuseconll:
        allckpt = torch.load("./T5PromptNER/bestckpt")
        model.promptnumber = allckpt["promptnumber"]
        model.promptembedding = allckpt["promptembedding"]
    else:
        promptnumber = 300
        taskname = "name entity recognition"
        promptembedding = getpromptembedding(model, tokenizer, promptnumber, taskname)
        model.set_prompt_embedding(promptnumber, promptembedding)

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
    output_dir = foldername + "tagger"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_optimizer_arguments = {"lr": learning_rate, "clip_threshold": max_grad_norm, "decay_rate": -0.8,
                                "weight_decay": weight_decay,
                                "scale_parameter": False, "relative_step": False}
    optimizer = Adafactor(params=filter(lambda p: p.requires_grad, model.parameters()), **base_optimizer_arguments)
    # distributed training
    model.train()

    startepoch = 0
    Best_F1 = 0.0

    logger.info("Begin train...")

    result_dict = {
        'epoch': [],
        'val_F1': [],
        'best_val_F1': Best_F1
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
