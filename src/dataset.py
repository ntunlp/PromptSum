import json
import random
import os
import sys

sys.path.append("..")
import pickle
import torch
import operator
from torch.utils.data import Sampler, Dataset, DataLoader
from datasets import load_dataset
import spacy
from tqdm import tqdm
from guidance import *
from rouge_score import rouge_scorer


class T5CNNDataset(Dataset):
    def __init__(self, dataset_args, args, tokenizer, split="train"):
        '''
        Args:
            dataset_args: e.g ["cnn_dailymail", '3.0.0']
            split: choice of ['train', 'validation', 'test']
        '''
        super(T5CNNDataset, self).__init__()
        self.args = args
        print("loading the dataset...")
        self.data = load_dataset(*dataset_args, cache_dir=args.dataset_cache_dir)
        if type(split) == str:
            self.data = self.data[split]
        else:
            self.data = self.data["train"]
            self.data = self.data.select(split)
            print("# Data points in this split: {}".format(len(split)))
        self.maxlen = args.max_length
        self.tokenizer = tokenizer
        self.num_entries = len(self.data)

        if args.guidance_type == "ents":
            self.spacy_nlp = spacy.load("en_core_web_sm")
            # spacy_ents_stats(self.data, self.spacy_nlp, args.ents_stats_max_len)
            if split.startswith("train") and args.build_ents_freq:
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
    def __init__(self, max_length, max_guidance_length, max_target_length, pad_token_id):
        self._max_length = max_length
        self._max_guidance_length = max_guidance_length
        self._max_target_length = max_target_length
        self._pad_token_id = pad_token_id

    def __call__(self, batch):

        sequences, targets, ents = list(zip(*batch))

        input_ids, attention_mask = self.pad_sequence(
            sequences,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )

        ents_ids, _ = self.pad_sequence(
            ents,
            max_sequence_length=self._max_guidance_length,
            pad_token_id=self._pad_token_id
        )

        target_ids, target_mask = self.pad_target(targets, max_sequence_length=self._max_target_length, pad_token_id=self._pad_token_id)

        output = input_ids, attention_mask, target_ids, target_mask, ents_ids
        return output

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

def get_dataloader(num_workers,dataset, batch_size, max_len, max_guidance_len, max_target_length, pad_id, sampler):
    collate_fn = SmartBatchingCollate(
        max_length=max_len,
        max_guidance_length=max_guidance_len,
        max_target_length=max_target_length,
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
