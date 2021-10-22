import json
import random
import os
import sys

sys.path.append("..")
import pdb
import re
import pdb
import math
import torch
import numpy as np
import linecache
from pathlib import Path


from collections import Counter
from torch.utils import data
import random
import numpy as np
import multiprocessing
import more_itertools

import torch
import torch.nn as nn
from torch.utils.data import Sampler, Dataset, DataLoader
from datasets import load_dataset
import spacy

class T5CNNDataset(Dataset):
    def __init__(self, dataset_args, maxlen, max_ent_len, tokenizer, split):
        '''
        Args:
            dataset_args: e.g ["cnn_dailymail", '3.0.0']
            split: choice of ['train', 'validation', 'test']
        '''
        super(T5CNNDataset, self).__init__()
        self.data = load_dataset(*dataset_args, split=split, cache_dir="/export/home/cache/")
        self.maxlen = maxlen
        self.max_ent_len = max_ent_len
        self.tokenizer = tokenizer
        self.num_entries = len(self.data)
        self.spacy_nlp = spacy.load("en_core_web_sm")


    def __getitem__(self, idx):
        inputdata = self.data[idx]['article']
        targetdata = self.data[idx]['highlights']
        ents = self.spacy_nlp(inputdata).ents
        input_entities = ','.join([ent.text for ent in ents]) # can decide which delimiter works the best, just pick comma first


        inputres = self.tokenizer.batch_encode_plus([inputdata], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        targetres = self.tokenizer.batch_encode_plus([targetdata], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        input_ents_res = self.tokenizer.batch_encode_plus([input_entities], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")

        return inputres["input_ids"].squeeze(), targetres["input_ids"].squeeze(), input_ents_res['input_ids'].squeeze()

    def __len__(self):
        return self.num_entries


class SmartBatchingCollate:
    def __init__(self, max_length, max_ent_length, pad_token_id):
        self._max_length = max_length
        self._max_ent_length = max_ent_length
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
            max_sequence_length=self._max_ent_length,
            pad_token_id=self._pad_token_id
        )

        target_ids, target_mask = self.pad_target(targets, max_sequence_length=self._max_length, pad_token_id=self._pad_token_id)

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

def get_dataloader(num_workers,dataset, batch_size, max_len, max_ent_len, pad_id, sampler):
    collate_fn = SmartBatchingCollate(
        max_length=max_len,
        max_ent_len=args.max_ent_len,
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
