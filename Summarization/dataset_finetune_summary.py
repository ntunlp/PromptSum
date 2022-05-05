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
from torch.utils.data import Sampler, Dataset, DataLoader
from rouge_score import rouge_scorer
import random
from nltk.corpus import stopwords

class T5SummarizationDataset(Dataset):
    def __init__(self, filename, split, maxlen, tokenizer, newtgentasktokens, answertoken, args, seed=0,
                 counterfactual_removal=False):
        super(T5SummarizationDataset, self).__init__()

        self.filename = filename
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.gentasktoken = newtgentasktokens
        self.answertoken = answertoken
        self.args = args
        self.save_path = args.few_shot_save_dir
        self.seed = seed

        self.data = []
        self.data = self.getalldata(self.filename)
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
                    entpath = f'{self.save_path}seed_{self.seed}/data_for_bert_{self.seed}/valident.txt'
                    self.allent = self.handleentfile(entpath)

        # counterfactual training
        self.counterfactual_removal = args.counterfactual_removal
        if self.counterfactual_removal:
            self.counterfactual_remove()
            print("# After augmenting, Data points in this split: {}".format(len(self.data)))

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

    def set_allent_for_valid(self):
        entpath = f'{self.save_path}seed_{self.seed}/data_for_bert_{self.seed}/T5valident.pkl'
        with open(entpath, "rb") as f:
            self.allent = pickle.load(f)

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
            onedata.append(linelist[0])
            onedata.append(linelist[1])
            if len(linelist) > 2:
                onedata.append(linelist[2])
            alldata.append(onedata)
        f.close()

        return alldata

    def set_tagger_tokenizer(self, tagger, tokenizer):
        self.tagger = tagger
        self.tagtokenizer = tokenizer

    def __getitem__(self, idx):
        inputdata = self.data[idx][0]
        targetdata = self.data[idx][1]

        # guidance
        input_guidance = "None"
        pred_guidance = "None"
        
        stop_words = set(stopwords.words('english'))
        # 1st option: based on entities
        if self.args.guidance_type == "ents":
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
                elif self.args.guidance_mode == "input_and_target":
                    ents_x = self.spacy_nlp(inputdata).ents
                    ents_x = [ent.text for ent in ents_x]
                    ents_y = self.spacy_nlp(targetdata).ents
                    ents_y = [ent.text for ent in ents_y]
                    ents_intersection = [ent for ent in ents_x if ent in ents_y]
                    ents_intersection = list(dict.fromkeys(ents_intersection))  # remove duplicates, while keeping order
                    if ents_intersection == []:
                        ents_intersection = ents_x[:max(2,len(ents_y))]
                elif self.args.guidance_mode == "input_salient_sents":
                    top_sents = self.find_salient_sents(inputdata, 5)
                    ents = self.spacy_nlp(top_sents).ents
                    ents = [ent.text for ent in ents]
                elif self.args.guidance_mode == "input_most_frequent":
                    ents = self.spacy_nlp(inputdata).ents
                    ents = [ent.text for ent in ents]
                    counts = {}
                    for ent in ents:
                        if not (ent in counts.keys()):
                            counts[ent] = 0
                        counts[ent] += 1
                    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
                    top_ents = []
                    for k in sorted_counts.keys():
                        top_ents.append(k)
                        if len(top_ents) >= 20:
                            break
                else:
                    ents = self.spacy_nlp(inputdata).ents
                    # for e in ents:
                    #     if e.text.lower() in stop_words:
                    #         print('e.text: ', e.text)
                    #         print('e.label_: ',e.label_)
                    #         raise Exception('end')
                    ents = [ent.text for ent in ents]
                input_guidance = self.args.separator.join(ents) # can decide which delimiter works the best, just pick comma first
            else: #use bert_tagger
                ####for train
                if self.split.startswith("train"):
                    tempdata = re.sub(' +', ' ', inputdata)
                    if tempdata in self.allent.keys():
                        input_guidance = self.allent[tempdata]
                    else:
                        print("we can not find inputdata in the dictionary!! There should be some errors!")
                else:
                    if self.args.guidance_mode == 'target':
                        tempdata = re.sub(' +', ' ', inputdata)
                        if tempdata in self.allent.keys():
                            input_guidance = self.allent[tempdata]
                        else:
                            print("we can not find inputdata in the dictionary!! There should be some errors!")
                    else:
                        # tempdata = re.sub(' +', ' ', inputdata)
                        # inputres = self.tagtokenizer.batch_encode_plus([tempdata], padding=True, max_length=self.maxlen, truncation=True, return_tensors="pt")
                        # input_ids = inputres["input_ids"].to(self.args.device)
                        # attention_mask = inputres["attention_mask"].to(self.args.device)
                        # input = {"input_ids": input_ids, "attention_mask":attention_mask}
                        # taginput,tagpreds = self.tagger._generative_step_for_tagger(input)
                        # allentitylist = tagpreds[0].split(',')
                        # if allentitylist == []:
                        #    allentitylist = ["none"]
                        #
                        # #input_guidance = self.args.separator.join(list(set(allentitylist)))
                        # input_guidance = self.args.separator.join(list(dict.fromkeys(allentitylist)))
                        # #input_guidance = self.args.separator.join(allentitylist)
                        #
                        # # inputents = self.spacy_nlp(inputdata).ents
                        # # inputents = [ent.text for ent in inputents]
                        # # inputents.extend(allentitylist)
                        # # input_guidance = self.args.separator.join(inputents)

                        tempdata = re.sub(' +', ' ', inputdata)
                        if tempdata in self.allent.keys():
                            input_guidance = self.allent[tempdata]
                        else:
                            print(
                                "For valid: we can not find inputdata in the dictionary!! There should be some errors!")

            # if counterfactual_removed, remove removed_ents in the input_guidance
            if self.counterfactual_removal:
                if self.removed_ents[idx] != None:
                    for ent in self.removed_ents:
                        print('input_guidance: ', input_guidance)
                        input_guidance = input_guidance.replace(ent.text, '')
                        print('input_guidance2: ', input_guidance)
                        print('removed_ents[idx]: ', self.removed_ents[idx])
                        raise Exception('end')

        # 2nd option: based on salient sentences
        elif self.args.guidance_type == "sents":
            salient_sents = self.find_salient_sents(inputdata, 1)
            input_guidance = salient_sents
        # print(inputdata, " ****** ", targetdata, " &&&&&& ", input_guidance)
        inputres = self.tokenizer.batch_encode_plus([inputdata], padding=False, max_length=self.maxlen, truncation=True,
                                                    return_tensors="pt")
        targetres = self.tokenizer.batch_encode_plus([targetdata], padding=False, max_length=self.maxlen,
                                                     truncation=True, return_tensors="pt")
        inputentsres = self.tokenizer.batch_encode_plus([input_guidance], padding=False, max_length=self.maxlen,
                                                        truncation=True, return_tensors="pt")
        predentsres = self.tokenizer.batch_encode_plus([pred_guidance], padding=False, max_length=self.maxlen,
                                                       truncation=True, return_tensors="pt")

        return inputres["input_ids"].squeeze(), targetres["input_ids"].squeeze(), inputentsres['input_ids'].squeeze(), \
               predentsres['input_ids'].squeeze()

    def __len__(self):

        return self.num_entries

    def counterfactual_remove(self):
        '''
        Function to add counterfactually removed instances to data
        input:
            data: list of (text, summary) tuples
        '''
        inputdata = [i[0] for i in self.data]
        targetdata = [i[1] for i in self.data]
        new_inputdata = inputdata
        new_targetdata = targetdata
        removed_ents = [None] * len(inputdata)
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        for i in range(len(inputdata)):
            ents_x = self.spacy_nlp(inputdata[i]).ents
            ents_x = [ent.text for ent in ents_x]
            ents_y = self.spacy_nlp(targetdata[i]).ents
            ents_y = [ent.text for ent in ents_y]
            ents_intersection = [ent for ent in ents_x if ent in ents_y]
            ents_intersection = list(dict.fromkeys(ents_intersection))
            # split summaries into sentences
            sents = tokenizer.tokenize(targetdata[i])
            if len(sents) > 1:
                for sent in sents:
                    # for each sentence, find its entities
                    ents = self.spacy_nlp(sent).ents
                    # if it's in the intersection list
                    removed = [ent.text for ent in ents if ent.text in ents_intersection]
                    # if this list is not empty
                    if len(removed) > 0:
                        # construct counterfactual example
                        new_targetdata.append(' '.join(sents).replace(sent, ''))
                        new_inputdata.append(inputdata[i])
                        removed_ents.append(removed)
        # change self.data
        self.data = [(new_inputdata[i], new_targetdata[i]) for i in range(len(new_inputdata))]
        self.removed_ents = removed_ents

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


class SmartBatchingCollate:
    def __init__(self, args, tokenizer, max_length, max_guidance_length, pad_token_id):
        self.args = args
        self.tokenizer = tokenizer
        self._max_length = max_length
        self._max_guidance_length = max_guidance_length
        self._pad_token_id = pad_token_id

    def __call__(self, batch):
        sequences, targets, ents, predents = list(zip(*batch))

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
        if "DID" in self.args.model:
            right = False
        ents_ids, ents_mask = self.pad_sequence(
            ents,
            max_sequence_length=self._max_guidance_length,
            pad_token_id=self._pad_token_id,
            right=right
        )
        if "DID" in self.args.model:
            sep_ids = torch.ones((ents_ids.shape[0], 1), dtype=torch.long, device=ents_ids.device) * \
                      self.tokenizer.encode("[SEP]")[0]
            ents_ids = torch.cat((ents_ids, sep_ids), 1)
            sep_mask = torch.ones((ents_ids.shape[0], 1), dtype=torch.long, device=ents_ids.device)
            ents_mask = torch.cat((ents_mask, sep_mask), 1)
        predents_ids, predents_mask = self.pad_sequence(
            predents,
            max_sequence_length=self._max_guidance_length,
            pad_token_id=self._pad_token_id,
            right=right
        )
        # print(input_ids.shape, target_ids.shape, ents_ids.shape, predents_ids.shape)
        output = input_ids, attention_mask, target_ids, target_mask, ents_ids, ents_mask, predents_ids, predents_mask

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
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)
        padded_sequences = []
        attention_masks = []
        attend, no_attend = 1, 0
        for sequence in sequence_batch:
            new_sequence = list(sequence[:max_len])

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


def convert_data_to_txt(train_data, new_train_path, args):
    all_train_texts, all_train_summaries = [], []
    for idx in range(len(train_data)):
        text = train_data[idx][args.text_key]
        text = " ".join(text.split("\n"))
        summary = train_data[idx][args.summary_key]
        summary = " ".join(summary.split("\n"))
        all_train_texts.append(text)
        all_train_summaries.append(summary)
    print("writing to: {}".format(new_train_path))
    with open(new_train_path, "w") as f:
        for idx in range(len(all_train_texts)):
            to_write = all_train_texts[idx] + "\t" + all_train_summaries[idx]
            if idx > 0:
                to_write = "\n" + to_write
            f.write(to_write)


def read_subsampled(args, tokenizer, allgentasktokens, answertoken, few_shot_seeds):
    '''
    This function reads in the few-shot datasets saved at save_path
    returns:
        list of tuples (train_dataset, valid_dataset)
    '''
    datasets = []
    for seed in few_shot_seeds:
        train_file_name = args.few_shot_save_dir + 'seed_{}/train.txt'.format(seed)
        valid_file_name = args.few_shot_save_dir + 'seed_{}/valid.txt'.format(seed)
        train_dataset = T5SummarizationDataset(train_file_name, "train", args.max_length, tokenizer, allgentasktokens, answertoken, args, seed,
                                               counterfactual_removal=args.counterfactual_removal)
        valid_dataset = T5SummarizationDataset(valid_file_name, "valid", args.max_length, tokenizer, allgentasktokens, answertoken, args, seed)
        datasets.append((train_dataset, valid_dataset, seed))

    return datasets


def subsample(dataset_args, args, tokenizer, few_shot_seeds):
    '''
    Function that subsamples a dataset and saves the results for few-shot exps
    args:
        few_shot_seeds: list of random seeds to produce the subsamples repeatively
    '''
    data = datasets.load_dataset(*dataset_args, cache_dir=args.dataset_cache_dir)
    train_data = data['train']
    valid_data = data['validation']
    len_train = len(train_data)
    len_valid = len(valid_data)
    for seed in few_shot_seeds:
        os.makedirs(args.few_shot_save_dir + 'seed_{}'.format(seed), exist_ok=True)
        # re-set random seed
        np.random.seed(seed)
        indices = np.random.choice(range(len_train), args.few_shot)
        train_data_new = train_data.select(indices)
        indices = np.random.choice(range(len_valid), args.few_shot)
        valid_data_new = valid_data.select(indices)
        # save
        train_path = args.few_shot_save_dir + 'seed_{}/train.txt'.format(seed)
        valid_path = args.few_shot_save_dir + 'seed_{}/valid.txt'.format(seed)
        convert_data_to_txt(train_data_new, train_path, args)
        convert_data_to_txt(valid_data_new, valid_path, args)
    # convert to original seed
    np.random.seed(args.seed)