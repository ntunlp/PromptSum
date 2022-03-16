import json
import random
import os
import sys
sys.path.append("..")
import pickle
import torch
import operator
import spacy
from torch.utils.data import Sampler, Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
from rouge_score import rouge_scorer
from guidance import *



class T5CNNDataset(Dataset):
    def __init__(self, dataset_args, args, tokenizer, split, data = None, subsample = False):
        '''
        Args:
            dataset_args: e.g ["cnn_dailymail", '3.0.0']
            split: choice of ['train', 'validation', 'test'] or indices marking each split 
            data: subsampled data, used for loading few-shot datasets
        '''
        super(T5CNNDataset, self).__init__()
        
        print("loading the dataset...")
        if subsample:
            self.data = data
        else:
            self.data = load_dataset(*dataset_args, data_dir = args.dataset_data_dir, cache_dir=args.dataset_cache_dir)
            if type(split) == str:
                self.data = self.data[split]
            else:
                self.data = self.data["train"]
                self.data = self.data.select(split)
        print("# Data points in this split: {}".format(len(self.data)))
        
        self.tokenizer = tokenizer
        self.args = args

        self.maxlen = args.max_length
        self.num_entries = len(self.data)

        if args.guidance_type == "ents":
            self.spacy_nlp = spacy.load("en_core_web_sm")
            if args.build_ents_freq and split.startswith("train"):
                print("building entities frequency...")
                self.ents_freq = spacy_build_ents_frequency(self.data, self.spacy_nlp, args.ents_freq_max_len)
                with open("ents_freq.pkl", "wb") as f:
                    pickle.dump(self.ents_freq, f)
            else:
                self.ents_freq = pickle.load(open("ents_freq.pkl", "rb"))
                print("loaded the entities frequency! There are {} entities".format(len(self.ents_freq.keys())))
        elif args.guidance_type == "sents":
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    def __getitem__(self, idx):
        inputdata = self.data[idx][self.args.text_key]
        targetdata = self.data[idx][self.args.summary_key]

        # guidance
        input_guidance = "None"
        # 1st option: based on entities
        if self.args.guidance_type == "ents":
            if self.args.guidance_mode == 'oracle':
                ents_x = self.spacy_nlp(inputdata).ents
                ents_x = [ent.text for ent in ents_x]
                ents_y = self.spacy_nlp(targetdata).ents
                ents_y = [ent.text for ent in ents_y]
                ents_intersection = [ent for ent in ents_x if ent in ents_y]
                ents_intersection = list(dict.fromkeys(ents_intersection)) # remove duplicates, while keeping order
                input_guidance = ','.join(ents_intersection)
            else:
                ents = self.spacy_nlp(inputdata).ents
                ents = [ent.text for ent in ents]
                if self.args.filter_ents_freq:
                    ents = [x for x in ents if x in self.ents_freq.keys() and self.ents_freq[x] >= self.args.min_ents_freq]
                input_guidance = ','.join(ents) # can decide which delimiter works the best, just pick comma first
        # 2nd option: based on salient sentences
        elif self.args.guidance_type == "sents":
            salient_sents = build_salient_sents(inputdata, targetdata, self.rouge_scorer, self.args)
            input_guidance = ' '.join(salient_sents)  # can decide which delimiter works the best, just pick comma first
        # print(input_guidance)

        inputres = self.tokenizer.batch_encode_plus([inputdata], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        targetres = self.tokenizer.batch_encode_plus([targetdata], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        input_ents_res = self.tokenizer.batch_encode_plus([input_guidance], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        
        return inputres["input_ids"].squeeze(), targetres["input_ids"].squeeze(), input_ents_res['input_ids'].squeeze()

    def __len__(self):
        
        return self.num_entries


class SmartBatchingCollate:
    def __init__(self, max_length, max_guidance_length, max_summary_length, pad_token_id):
        self._max_length = max_length
        self._max_guidance_length = max_guidance_length
        self._max_summary_length = max_summary_length
        self._pad_token_id = pad_token_id

    def __call__(self, batch):
        sequences, targets, ents = list(zip(*batch))
        input_ids, attention_mask = self.pad_sequence(
            sequences,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )
        ents_ids, ents_mask = self.pad_sequence(
            ents,
            max_sequence_length=self._max_guidance_length,
            pad_token_id=self._pad_token_id
        )
        target_ids, target_mask = self.pad_target(
            targets, 
            max_sequence_length=self._max_summary_length, 
            pad_token_id=self._pad_token_id
        )
        output = input_ids, attention_mask, target_ids, target_mask, ents_ids, ents_mask
        
        return output

    def pad_sequence(self, sequence_batch, max_sequence_length, pad_token_id):
        ##tokenize sequence_batch
        lens = []
        for sequence in sequence_batch:
            if len(sequence.shape) > 0:
                lens.append(sequence.shape[0])
            else:
                lens.append(0)
        max_batch_len = max(lens)
        max_len = min(max_batch_len, max_sequence_length)
        padded_sequences = []
        attention_masks = []
        attend, no_attend = 1, 0
        for sequence in sequence_batch:
            # As discussed above, truncate if exceeds max_len
            if len(sequence.shape) > 0:
                new_sequence = list(sequence[:max_len])
                attention_mask = [attend] * len(new_sequence)
                pad_length = max_len - len(new_sequence)
                new_sequence.extend([pad_token_id] * pad_length)
                attention_mask.extend([no_attend] * pad_length)
            else:
                new_sequence = [pad_token_id] * max_len
                attention_mask = [no_attend] * max_len
            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)

        padded_sequences = torch.tensor(padded_sequences)
        attention_masks = torch.tensor(attention_masks)
       
        return padded_sequences, attention_masks

    def pad_target(self, sequence_batch, max_sequence_length, pad_token_id):
        ##tokenize sequence_batch
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)    ####whether because max_length is not 512?
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
        
        return padded_sequences,attention_masks


def get_dataloader(num_workers,dataset, batch_size, max_length, max_guidance_length, max_summary_length, pad_id, sampler):
    collate_fn = SmartBatchingCollate(
        max_length=max_length,
        max_guidance_length=max_guidance_length,
        max_summary_length=max_summary_length,
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


def subsample(dataset_args, args, tokenizer, few_shot_seeds, save_path):
    '''
    Function that subsamples a dataset and saves the results for few-shot exps
    args:
        few_shot_seeds: list of random seeds to produce the subsamples repeatively
    '''
    train_dataset = T5CNNDataset(dataset_args, args, tokenizer, split='train')
    valid_dataset = T5CNNDataset(dataset_args, args, tokenizer, split='validation')
    len_train = len(train_dataset.data)
    len_valid = len(valid_dataset.data)
    for seed in few_shot_seeds:
        # re-set random seed
        np.random.seed(seed)
        indices = np.random.choice(range(len_train), args.few_shot)
        train_data_new = train_dataset.data.select(indices)
        indices = np.random.choice(range(len_valid), args.few_shot)
        valid_data_new = valid_dataset.data.select(indices)
        # save
        handler_train = open(f'{save_path}{args.few_shot}_few_shot_train_seed_{seed}',"wb")
        handler_valid = open(f'{save_path}{args.few_shot}_few_shot_valid_seed_{seed}',"wb")
        pickle.dump(train_data_new, handler_train)
        pickle.dump(valid_data_new, handler_valid)
        handler_train.close()
        handler_valid.close()
    # convert to original seed
    np.random.seed(args.seed)


def read_subsampled(dataset_args, args, few_shot_seeds, tokenizer, save_path):
    '''
    This function reads in the few-shot datasets saved at save_path
    returns:
        list of tuples (train_dataset, valid_dataset)
    '''
    datasets = []
    for seed in few_shot_seeds:
        handler_train = open(f'{save_path}{args.few_shot}_few_shot_train_seed_{seed}',"rb")
        handler_valid = open(f'{save_path}{args.few_shot}_few_shot_valid_seed_{seed}',"rb")
        train_data = pickle.load(handler_train)
        valid_data = pickle.load(handler_valid)
        handler_train.close()
        handler_valid.close()
        train_dataset = T5CNNDataset(dataset_args, args, tokenizer, 'train', data = train_data, subsample = True)
        valid_dataset = T5CNNDataset(dataset_args, args, tokenizer, 'valid', data = valid_data, subsample = True)
        datasets.append((train_dataset, valid_dataset))
    
    return datasets









    