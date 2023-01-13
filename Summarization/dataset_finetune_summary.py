from ast import excepthandler
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
import pickle
import random

from tqdm import tqdm
from torch.utils.data import Sampler, Dataset, DataLoader
from rouge_score import rouge_scorer
from nltk.corpus import stopwords



class SummarizationDataset(Dataset):
    def __init__(self, filename, split, maxlen, tokenizer, newtgentasktokens, answertoken, args, seed=0, save_path=None, human_eval=False):
        super(SummarizationDataset, self).__init__()

        self.filename = filename
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.gentasktoken = newtgentasktokens
        self.answertoken = answertoken
        self.args = args
        if save_path != None:
            self.save_path = save_path
        else:
            self.save_path = args.few_shot_save_dir
        self.seed = seed

        self.data = []
        self.data = self.getalldata(self.filename)
        #self.data = self.data[:20]
        if human_eval:
            init_p = pickle.load(open("human_eval_permutations/init_{}.pkl".format(args.dataset_name), "rb"))
            self.data = [self.data[x] for x in init_p]
            new_p = pickle.load(open("human_eval_permutations/{}.pkl".format(args.dataset_name), "rb"))
            self.data = [self.data[x] for x in new_p]
        else:
            p = np.random.permutation(len(self.data))
            with open("human_eval_permutations/init_{}.pkl".format(args.dataset_name), "wb") as f:
                pickle.dump(p, f)
            self.data = [self.data[x] for x in p]
        print("permuted self.data")
        self.num_entries = len(self.data)

        self.split = split
        self.tagger = None
        self.tagtokenizer = None

        self.allent = {}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        if args.use_t5_tagger:
            ####train valid test
            if self.split.startswith("train"):
                ####load entity file for training data
                entpath = f'{self.save_path}seed_{self.seed}/data_for_bert_{self.seed}/trainent.txt'
                self.allent = self.handleentfile(entpath)
            else:
                if self.args.guidance_mode == 'target':
                    # entpath = f'{self.save_path}seed_{self.seed}/data_for_bert_{self.seed}/valident.txt'
                    entpath = f'{self.save_path}seed_{0}/data_for_bert_{0}/valident.txt'
                    self.allent = self.handleentfile(entpath)

    def handleentfile(self, entpath):
        fe = open(entpath, 'r')
        allres = {}
        while True:
            oneline = fe.readline().strip()
            if not oneline:
                break
            content = oneline.split("\t")
            if len(content) != 2:
                print("train data entity error!!!!")
                continue
            doc = content[0]
            entlist = content[1]
            allres[doc] = entlist
        fe.close()
        return allres

    def set_allent_for_valid(self, entpath):
        # entpath = f'tagger_ckpt/{self.args.dataset}/{self.args.few_shot}/seed_{self.seed}/T5valident.pkl'
        print("entpath: ",entpath)
        with open(entpath, "rb") as f:
            self.allent = pickle.load(f)

    def check_entity_keys(self):
        not_in = 0
        for i in tqdm(range(len(self.data))):
            inputdata = self.data[i][0]
            tempdata = re.sub(' +', ' ', inputdata).strip()
            if not(tempdata in self.allent.keys()):
                not_in += 1
        print("{} Text entries not in the entity dictionary: {} (SHOULD BE 0!!!)".format(self.split, not_in))

    def getalldata(self, filename):
        f = open(filename, 'r')
        alldata = []
        i = 0
        while True:
            oneline = f.readline().strip()
            if not oneline:
                break
            linelist = oneline.split("\t")
            i += 1
            onedata = []
            try:
                onedata.append(linelist[0])
                onedata.append(linelist[1])
                if len(linelist) > 2:
                    onedata.append(linelist[2])
                alldata.append(onedata)
            except:
                continue
        f.close()
        return alldata

    def set_tagger_tokenizer(self, tagger, tokenizer):
        self.tagger = tagger
        self.tagtokenizer = tokenizer

    def shuffle_and_subsample(self):
        p = np.random.permutation(len(self.data))
        print(p[:50])
        with open("human_eval_permutations/{}.pkl".format(self.args.dataset_name), "wb") as f:
            pickle.dump(p, f)
            print("saved the permutation")
        p = p[:self.args.max_test_size]
        self.data = [self.data[x] for x in p]
        self.num_entries = len(self.data)
        print("\nShuffled and subsampled!")
        print("First data point:")
        print(self.data[0])

    def __getitem__(self, idx):
        inputdata = self.data[idx][0]
        targetdata = self.data[idx][1]
        # guidance
        input_guidance = "None"
        
        stop_words = set(stopwords.words('english'))

        if not self.args.use_t5_tagger:
            if self.args.guidance_mode == "target":
                ents = self.spacy_nlp(targetdata).ents
                ents = [ent.text for ent in ents]
                if ents == []:
                    ents = ["none"]
            if "target_unique" in self.args.guidance_mode:
                old_ents = self.spacy_nlp(targetdata).ents
                if 'filter' in self.args.guidance_mode:
                    if self.args.filter_type!=None:
                        old_ents = [ent.text for ent in old_ents if ent.label_ != self.args.filter_type]
                    else:
                        old_ents = [ent.text for ent in old_ents]
                    old_ents = [w for w in old_ents if not w.lower() in stop_words]
                else:
                    old_ents = [ent.text for ent in old_ents]
                # remove entities case-insensitively
                marker = set()
                ents = []
                for l in old_ents:
                    ll = l.lower()
                    if ll not in marker:  # test presence
                        marker.add(ll)
                        ents.append(l)
                if len(ents) == 0:
                    ents_x = self.spacy_nlp(inputdata).ents
                    ents_x = [ent.text for ent in ents_x]
                    ents = ents_x[:2]
                if "shuffle" in self.args.guidance_mode:
                    # shuffle ents
                    random.shuffle(ents, random.random)
            else:
                ents = self.spacy_nlp(inputdata).ents
                ents = [ent.text for ent in ents]
            # input_guidance = self.args.separator.join(ents) # can decide which delimiter works the best, just pick comma first
            input_guidance = ','.join(list(dict.fromkeys(ents)))
        else: #use bert_tagger
            ####for train
            if self.split.startswith("train"):
                tempdata = re.sub(' +', ' ', inputdata).strip()
                if tempdata in self.allent.keys():
                    input_guidance = self.allent[tempdata]
                else:
                    print("We can not find TRANING inputdata in the dictionary!! There should be some errors!")
            else:
                if self.args.guidance_mode == 'target':
                    tempdata = re.sub(' +', ' ', inputdata).strip()
                    if tempdata in self.allent.keys():
                        input_guidance = self.allent[tempdata]
                    else:
                        print("We can not find VALIDATION - TARGET inputdata in the dictionary!! There should be some errors!")
                else:
                    tempdata = re.sub(' +', ' ', inputdata).strip()
                    if tempdata in self.allent.keys():
                        input_guidance = self.allent[tempdata]
                    else:
                        print("For valid: we can not find VALIDATION - PREDICTION inputdata in the dictionary!! There should be some errors!")
        # after finding it, I remove the 'REMOVEDN' argument:
        inputdata = re.sub('REMOVED[0-9]+ ', '', inputdata)

        if len(targetdata.split()) == 0:
            targetdata = "None"
        inputres = self.tokenizer.batch_encode_plus([inputdata], padding=False, max_length=self.maxlen, truncation=True,  return_tensors="pt")
        targetres = self.tokenizer.batch_encode_plus([targetdata], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        inputentsres = self.tokenizer.batch_encode_plus([input_guidance], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        
        return inputres["input_ids"].squeeze(), targetres["input_ids"].squeeze(), inputentsres['input_ids'].squeeze(dim=0)

    def __len__(self):

        return self.num_entries

    def find_salient_sents(self, text, n):
        sents = nltk.sent_tokenize(text)
        r1s = []
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        for j in range(len(sents)):
            sent = sents[j]
            rest = " ".join(sents[:j] + sents[(j + 1):])
            rouge_scores = scorer.score(rest, sent)
            r1 = rouge_scores["rouge1"].fmeasure
            r1s.append(r1)
        idx = np.argsort(np.array(r1s))[::-1]
        top_idx = idx[:n]
        top_idx.sort()
        top_sents = [sents[i] for i in top_idx]
        top_sents = " ".join(top_sents)

        return top_sents

class T5SummarizationDatasetForControlGen(Dataset):
    def __init__(self, filename, split, maxlen, tokenizer, newtgentasktokens, answertoken, args, seed=0, save_path=None):
        super(T5SummarizationDatasetForControlGen, self).__init__()

        self.filename = filename
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.gentasktoken = newtgentasktokens
        self.answertoken = answertoken
        self.args = args
        if save_path != None:
            self.save_path = save_path
        else:
            self.save_path = args.few_shot_save_dir
        self.seed = seed

        self.data = []
        self.num_entries = len(self.data)

        self.split = split

        self.tagger = None
        self.tagtokenizer = None

        self.allent = [] # same idx as self.data
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    def set_tagger_tokenizer(self, tagger, tokenizer):
        self.tagger = tagger
        self.tagtokenizer = tokenizer

    def __getitem__(self, idx):
        inputdata = self.data[idx][0]
        targetdata = self.data[idx][1]
        ents = self.allent[idx]

        # guidance
        input_guidance = "None"
        
        stop_words = set(stopwords.words('english'))
        # 1st option: based on entities
        if self.args.guidance_type == "ents":
            input_guidance = ents
            inputdata = re.sub('REMOVED[0-9]+ ', '', inputdata)

        inputres = self.tokenizer.batch_encode_plus([inputdata], padding=False, max_length=self.maxlen, truncation=True,  return_tensors="pt")
        targetres = self.tokenizer.batch_encode_plus([targetdata], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        inputentsres = self.tokenizer.batch_encode_plus([input_guidance], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")

        return inputres["input_ids"].squeeze(), targetres["input_ids"].squeeze(), inputentsres['input_ids'].squeeze()

    def __len__(self):

        return self.num_entries


class SmartBatchingCollate:
    def __init__(self, args, tokenizer, max_length, max_guidance_length, pad_token_id):
        self.args = args
        self.tokenizer = tokenizer
        self._max_length = max_length
        self._max_guidance_length = max_guidance_length
        self._pad_token_id = pad_token_id

    def __call__(self, batch):
        sequences, targets, ents = list(zip(*batch))
        # input
        input_ids, attention_mask = self.pad_sequence(
            sequences,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )
        # target
        target_ids, target_mask = self.pad_target(
            targets,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )
        # guidance
        right = True
        ents_ids = target_ids
        ents_mask = target_mask
        if self.args.model in ["T5MixPrompt", "PegasusMixPrompt", "CTRLsum"]:
            try:
                # import pdb;pdb.set_trace()
                ents_ids, ents_mask = self.pad_sequence(
                    ents,
                    max_sequence_length=self._max_guidance_length,
                    pad_token_id=self._pad_token_id,
                    right=right
                )
            except:
                print(f'batch: {batch}')
                print(f'ents: {ents}')
                raise Exception('end')
        
        output = input_ids, attention_mask, target_ids, target_mask, ents_ids, ents_mask

        return output

    def pad_target(self, sequence_batch, max_sequence_length, pad_token_id):
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)
        padded_sequences = []
        attention_masks = []
        attend, no_attend = 1, 0
        for sequence in sequence_batch:
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

    def pad_sequence(self, sequence_batch, max_sequence_length, pad_token_id, right=True):
        # try:
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        # except TypeError:
        #     max_batch_len = 1
        max_len = min(max_batch_len, max_sequence_length)
        padded_sequences = []
        attention_masks = []
        attend, no_attend = 1, 0
        for sequence in sequence_batch:
            try:
                new_sequence = list(sequence[:max_len])
            except:
                #0-d tensor
                new_sequence = [sequence.item()]

            attention_mask = [attend] * len(new_sequence)
            pad_length = max_len - len(new_sequence)

            if right:
                new_sequence.extend([pad_token_id] * pad_length)
                attention_mask.extend([no_attend] * pad_length)
            else:
                padding = [pad_token_id] * pad_length
                new_sequence = padding + new_sequence
                padding = [no_attend] * pad_length
                attention_mask = padding + attention_mask

            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)

        padded_sequences = torch.tensor(padded_sequences)
        attention_masks = torch.tensor(attention_masks)

        return padded_sequences, attention_masks


def read_subsampled(tokenizer, allgentasktokens, answertoken, few_shot_seeds, args):
    '''
    This function reads in the few-shot datasets saved at save_path
    returns:
        list of tuples (train_dataset, valid_dataset)
    '''
    datasets = []
      
    for seed in few_shot_seeds:
        train_file_name = args.few_shot_save_dir + 'seed_{}/train.txt'.format(seed)
        train_dataset = SummarizationDataset(train_file_name, "train", args.max_length, tokenizer, allgentasktokens, answertoken, args, seed)
        valid_file_name = args.few_shot_save_dir + 'seed_{}/valid.txt'.format(seed)
        valid_dataset = SummarizationDataset(valid_file_name, "valid", args.max_length, tokenizer, allgentasktokens, answertoken, args, seed)
        datasets.append((train_dataset, valid_dataset, seed))

    return datasets


