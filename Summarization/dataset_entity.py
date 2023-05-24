import sys
sys.path.append("../..")

import spacy
import torch
import datasets
import os
import numpy as np
import nltk
import nltk
import random
import re
import gc
from tqdm import tqdm

from torch.utils.data import Sampler, Dataset, DataLoader, SequentialSampler
from rouge_score import rouge_scorer
from utils import *
#from transformers import BertTokenizer, T5Tokenizer, T5ForConditionalGeneration

from dataset_pretrain import *
from models.model_entity import ModelEntity



def get_data(few_shot_seeds, save_path, args):
    usetrain, usevalid, usetest = True, True, True
    spacy_nlp = spacy.load("en_core_web_sm")
    alltrainfile, allvalidfile, alltestfile = [], [], []
    # few_shot_seeds = [0]
    i = 0
    for seed in few_shot_seeds:
        # if i > 0:
        #     break # should generate for all seeds
        i += 1
        train_file_name = save_path + 'seed_{}/train.txt'.format(seed)
        valid_file_name = save_path + 'seed_{}/valid.txt'.format(seed)
        if args.full_testset or args.big_testset:
            test_file_name = save_path + 'seed_{}/test.txt'.format(seed)
        else:
            if args.few_shot == "full":
                args.max_test_size = args.max_val_size
            else:
                args.max_test_size = int(args.few_shot)
            test_file_name = valid_file_name
        handler_train = open(train_file_name, "r")
        handler_valid = open(valid_file_name, "r")
        handler_test = open(test_file_name, "r")

        alldoc, allsum = [], []
        count = 0
        if usetrain:
            while True:
                oneline = handler_train.readline().strip()
                if not oneline:
                    break
                onedata = oneline.split("\t")
                if len(onedata) != 2:
                    print("train doc sum split error")
                    continue
                onedoc = re.sub(' +', ' ', onedata[0].replace("\n"," "))
                alldoc.append(onedoc)
                onesum = re.sub(' +', ' ', onedata[1].replace("\n"," "))
                allsum.append(onesum)
                count += 1
                if count % 10000 == 0:
                    print("TRAIN - did {}".format(count))
        if usevalid:
            count = 0
            while True:
                oneline = handler_valid.readline().strip()
                if not oneline:
                    break
                onedata = oneline.split("\t")
                if len(onedata) != 2:
                    print("valid doc sum split error")
                    continue
                onedoc = re.sub(' +', ' ', onedata[0].replace("\n", " "))
                alldoc.append(onedoc)
                onesum = re.sub(' +', ' ', onedata[1].replace("\n", " "))
                allsum.append(onesum)
                count += 1
                if count % 1000 == 0:
                    print("VALID - did {}".format(count))
        if usetest:
            count = 0
            while True:
                oneline = handler_test.readline().strip()
                if not oneline:
                    break
                onedata = oneline.split("\t")
                if len(onedata) != 2:
                    print("test doc sum split error")
                    continue
                onedoc = re.sub(' +', ' ', onedata[0].replace("\n", " "))
                alldoc.append(onedoc)
                onesum = re.sub(' +', ' ', onedata[1].replace("\n", " "))
                allsum.append(onesum)
                count += 1
                if count % 1000 == 0:
                    print("TEST - did {}".format(count))

        handler_train.close()
        handler_valid.close()
        handler_test.close()
        doc_sum_path = f'{save_path}seed_{seed}/data_for_bert_{seed}/'
        if not os.path.exists(doc_sum_path):
            os.makedirs(doc_sum_path, exist_ok=True)

        ##### seperate it to document + summary
        docpath = doc_sum_path + "doc.txt"
        sumpath = doc_sum_path + "sum.txt"
        f = open(docpath, 'w')
        for oned in alldoc:
            f.write(oned + "\n")
        f.close()
        f = open(sumpath, 'w')
        for ones in allsum:
            f.write(ones + "\n")
        f.close()

        #### get train and valid data for bert tagger
        docwithlabel_train, docwithlabel_valid, docwithlabel_test = get_train_valid_data(sumpath, docpath, doc_sum_path, spacy_nlp, args)
        alltrainfile.append(docwithlabel_train)
        allvalidfile.append(docwithlabel_valid)
        alltestfile.append(docwithlabel_test)

    return alltrainfile, allvalidfile, alltestfile


def get_train_valid_data(sumpath, docpath, doc_sum_path, spacy_nlp, args):
    #### get predict label of summarization
    sum_y_pred = get_predict_label_for_sum(doc_sum_path, sumpath, spacy_nlp, args)

    #### get label for document
    alldocandlabel, allentityfortrain, allentityforvalid, allentityfortest = get_doc_label(sum_y_pred, docpath, args)

    #### split to train and valid
    docwithlabeltrain, docwithlabelvalid, docwithlabeltest = get_train_valid(
        alldocandlabel, doc_sum_path, allentityfortrain, allentityforvalid, allentityfortest, args)

    return docwithlabeltrain, docwithlabelvalid, docwithlabeltest


def get_predict_label_for_sum(doc_sum_path, sumpath, spacy_nlp, args):
    #####handle sumfile to fake conll format and use NER model to label it
    allpreds = []
    if not args.if_spacy:
        print("Finding entities with T5 tagger...")
        sumwithfakelabel = doc_sum_path + "sumwithfakelabel.txt"
        allsumwithfakelabeldata = getfilewithlabel(sumpath, sumwithfakelabel)
        model_name = args.model_name
        t5model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir = args.cache_path)
        tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir = args.cache_path)
        model = T5forFinetuneEntity(t5model, tokenizer, args)
        test_dataset = DatasetPretrainEntity(sumwithfakelabel, 512, tokenizer)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = get_dataloader_tag(4, test_dataset, 8, 512, test_dataset.tokenizer.pad_token_id, test_sampler)

        allckpt = torch.load("../support_files/conll_bestckpt")
        model.promptnumber = allckpt["promptnumber"]
        model.promptembedding = allckpt["promptembedding"]

        model.to(args.device)
        model.eval()

        with torch.no_grad():
            for step, batch in tqdm(enumerate(test_dataloader)):
                inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                          "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
                sen, target, preds = model._generative_step(inputs)
                allpreds.extend(preds)

        torch.cuda.empty_cache()
        del model, tokenizer, test_dataloader
        gc.collect()

        assert len(allpreds) == len(allsumwithfakelabeldata)
    else:
        print("Finding entities with spacy...")
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
            if index % 1000 == 0:
                print(index)
        fin.close()

    return allpreds


def getfilewithlabel(file, filewithfakelabel):
    fin = open(file,'r')
    alldata = []
    while True:
        oneline = fin.readline().strip()
        if not oneline:
            break
        alldata.append(oneline)
    fin.close()

    fo = open(filewithfakelabel, 'w')

    for onedata in alldata:
        fo.write(onedata+"\tend\n")
    fo.close()

    return alldata


def get_doc_label(sum_y_pred, docfile, args):
    alldocres, resfortrain, resforvalid, resfortest = getdocandent(docfile, sum_y_pred, args)

    allentityfortrain = []
    for i in tqdm(range(len(resfortrain))):
        onedata = resfortrain[i].split('\t')
        onedoc = onedata[0]
        oneent = onedata[1]
        allentityfortrain.append([onedoc, oneent])

    allentityforvalid = []
    for i in tqdm(range(len(resforvalid))):
        onedata = resforvalid[i].split('\t')
        onedoc = onedata[0]
        oneent = onedata[1]
        allentityforvalid.append([onedoc, oneent])

    allentityfortest = []
    for i in tqdm(range(len(resfortest))):
        onedata = resfortest[i].split('\t')
        onedoc = onedata[0]
        oneent = onedata[1]
        allentityfortest.append([onedoc, oneent])

    alldocandlabel = []
    for i in tqdm(range(len(alldocres))):
        onedata = alldocres[i].split('\t')
        onedoc = onedata[0]
        oneent = onedata[1]
        alldocandlabel.append([onedoc, oneent])

    return alldocandlabel, allentityfortrain, allentityforvalid, allentityfortest


def getdocandent(docfile, sum_y_pred, args):
    f = open(docfile,'r')
    alldoc = []
    while True:
        oneline = f.readline().strip()
        if not oneline:
            break
        alldoc.append(oneline)
    f.close()
    resfortrain, resforvalid, resfortest = [], [], []
    if args.few_shot == "full":
        trainsize = len(alldoc) - args.max_val_size - args.max_test_size
        valsize = args.max_val_size
    else:
        trainsize = (len(alldoc) - args.max_test_size) // 2
        valsize = trainsize
    print("alldoc", len(alldoc))
    print("train size: {}, valsize: {}".format(trainsize, valsize))
    #trainsize = 100
    allres = []
    for i in tqdm(range(len(alldoc))):
        if i < trainsize:
            resfortrain.append(alldoc[i] + "\t" + sum_y_pred[i])
        elif (i >= trainsize) and (i < (trainsize + valsize)):
            resforvalid.append(alldoc[i] + "\t" + sum_y_pred[i])
        else:
            resfortest.append(alldoc[i] + "\t" + sum_y_pred[i])
        allres.append(alldoc[i] + "\t" + sum_y_pred[i])

    return allres, resfortrain, resforvalid, resfortest


def get_train_valid(alldocandlabel, doc_sum_path, allentityfortrain, allentityforvalid, allentityfortest, args):
    docwithlabel_train = doc_sum_path + "docwithlabel_train.txt"
    docwithlabel_valid = doc_sum_path + "docwithlabel_valid.txt"
    docwithlabel_test = doc_sum_path + "docwithlabel_test.txt"

    fout = open(docwithlabel_train, 'w')
    fout_1 = open(docwithlabel_valid, 'w')
    fout_2 = open(docwithlabel_test, 'w')

    if args.few_shot == "full":
        trainsize = len(alldocandlabel) - args.max_val_size - args.max_test_size
        valsize = args.max_val_size
    else:
        trainsize = (len(alldocandlabel) - args.max_test_size) // 2
        valsize = trainsize
    for aa in tqdm(range(len(alldocandlabel))):
        onedata = alldocandlabel[aa]
        if aa < trainsize:
            fout.write(onedata[0] + "\t" + onedata[1] + "\n")
        elif (aa >= trainsize) and (aa < (trainsize + valsize)):
            fout_1.write(onedata[0] + "\t" + onedata[1] + "\n")
        else:
            fout_2.write(onedata[0] + "\t" + onedata[1] + "\n")
    fout.close()
    fout_1.close()
    fout_2.close()

    ### save train ent
    train_ent = doc_sum_path + "trainent.txt"
    fe = open(train_ent, 'w')
    for i in range(len(allentityfortrain)):
        if allentityfortrain[i][1] != []:
            fe.write(allentityfortrain[i][0].strip() + "\t" + allentityfortrain[i][1] + '\n')
        else:
            fe.write(allentityfortrain[i][0].strip() + "\tnone\n")
    fe.close()

    ### save valid ent
    valid_ent = doc_sum_path + "valident.txt"
    fe = open(valid_ent, 'w')
    for i in range(len(allentityforvalid)):
        if allentityforvalid[i][1] != []:
            fe.write(allentityforvalid[i][0].strip() + "\t" + allentityforvalid[i][1] + '\n')
        else:
            fe.write(allentityforvalid[i][0].strip() + "\tnone\n")
    fe.close()

    ### save test ent
    test_ent = doc_sum_path + "testent.txt"
    fe = open(test_ent, 'w')
    for i in range(len(allentityfortest)):
        if allentityfortest[i][1] != []:
            fe.write(allentityfortest[i][0].strip() + "\t" + allentityfortest[i][1] + '\n')
        else:
            fe.write(allentityfortest[i][0].strip() + "\tnone\n")
    fe.close()

    return docwithlabel_train, docwithlabel_valid, docwithlabel_test


def get_dataloader_tag(num_workers, dataset, batch_size, max_len, pad_id, sampler):
    collate_fn = SmartBatchingCollateTag(
        max_length=max_len,
        pad_token_id=pad_id
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        # shuffle=True, #####?????
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


class SmartBatchingCollateTag:
    def __init__(self, max_length, pad_token_id):
        self._max_length = max_length
        self._pad_token_id = pad_token_id

    def __call__(self, batch):

        sequences, targets = list(zip(*batch))

        input_ids, attention_mask = self.pad_sequence(
            sequences,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )

        target_ids, target_mask = self.pad_target(targets, max_sequence_length=self._max_length, pad_token_id=self._pad_token_id)

        output = input_ids, attention_mask, target_ids, target_mask

        return output

    def pad_target(self, sequence_batch, max_sequence_length, pad_token_id):
        ##tokenize sequence_batch
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)  ####whether because max_length is not 512?
        padded_sequences = []
        attention_masks = []
        attend, no_attend = 1, 0
        for sequence in sequence_batch:
            # As discussed above, truncate if exceeds max_len
            new_sequence = list(sequence[:max_len])
            attention_mask = [attend] * len(new_sequence)
            pad_length = max_len - len(new_sequence)
            new_sequence.extend([pad_token_id] * pad_length)
            attention_mask.extend([no_attend] * pad_length)
            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)
        padded_sequences = torch.tensor(padded_sequences)
        attention_masks = torch.tensor(attention_masks)

        return padded_sequences, attention_masks

    def pad_sequence(self, sequence_batch, max_sequence_length, pad_token_id):
        ##tokenize sequence_batch
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)
        padded_sequences = []
        attention_masks = []
        attend, no_attend = 1, 0
        for sequence in sequence_batch:
            # As discussed above, truncate if exceeds max_len
            new_sequence = list(sequence[:max_len])

            attention_mask = [attend] * len(new_sequence)
            pad_length = max_len - len(new_sequence)

            new_sequence.extend([pad_token_id] * pad_length)
            attention_mask.extend([no_attend] * pad_length)

            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)

        padded_sequences = torch.tensor(padded_sequences)
        attention_masks = torch.tensor(attention_masks)

        return padded_sequences, attention_masks


