import sys
import torch
import random
import pickle5

from torch.utils.data import Sampler, Dataset, DataLoader



class T5DatasetPretrainConll(Dataset):
    def __init__(self, filename, maxlen, tokenizer):
        super(T5DatasetPretrainConll, self).__init__()
        self.filename = filename
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.data = []
        self.data = self.getalldata(self.filename)
        self.num_entries = len(self.data)

    def getalldata(self,filename):
        f = open(filename,'r')
        alldata = []
        errnum = 0
        while True:
            oneline = f.readline().strip()
            if not oneline:
                break
            linelist = oneline.split("\t")
            if len(linelist) != 2:
                #print(oneline)
                errnum += 1
                continue
            onedata = []
            onedata.append(linelist[0])
            onedata.append(linelist[1])
            alldata.append(onedata)
        f.close()
        #print(errnum)
        return alldata

    def __getitem__(self, idx):
        inputdata = self.data[idx][0]
        targetdata = self.data[idx][1]
        inputres = self.tokenizer.batch_encode_plus([inputdata], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        targetres = self.tokenizer.batch_encode_plus([targetdata], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        #if idx < 10:
        #    print(inputres)
        #    print(targetres)
        #    sys.stdout.flush()

        return inputres["input_ids"].squeeze(), targetres["input_ids"].squeeze()

    def __len__(self):

        return self.num_entries


class T5DatasetPretrain(Dataset):
    def __init__(self, texts, ents, target, maxlen, tokenizer, args):
        super(T5DatasetPretrain, self).__init__()
        self.texts = texts
        self.ents = ents
        self.target = target
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.args = args

        self.num_entries = len(self.texts)

    def __getitem__(self, idx):
        inputdata = self.texts[idx]
        if len(inputdata) == 0:
            inputdata = "empty"

        targetdata = self.target[idx]
        if len(targetdata) == 0:
            targetdata = "empty"

        entitydata = self.ents[idx]
        if len(entitydata) == 0:
            entitydata = ["none"]
        entitydata = self.args.separator.join(entitydata)

        inputres = self.tokenizer.batch_encode_plus([inputdata], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        targetres = self.tokenizer.batch_encode_plus([targetdata], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        entityres = self.tokenizer.batch_encode_plus([entitydata], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        
        return inputres["input_ids"].squeeze(), targetres["input_ids"].squeeze(), entityres["input_ids"].squeeze(dim=0)

    def __len__(self):

        return self.num_entries


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

        target_ids, target_mask = self.pad_target(targets,max_sequence_length=self._max_length,pad_token_id=self._pad_token_id)

        output = input_ids, attention_mask, target_ids, target_mask

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


class SmartBatchingCollateTagPretrain:
    def __init__(self, max_length, pad_token_id):
        self._max_length = max_length
        self._pad_token_id = pad_token_id

    def __call__(self, batch):

        sequences, targets, entities = list(zip(*batch))

        input_ids, attention_mask = self.pad_sequence(
            sequences,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )

        #target_ids, target_mask = self.pad_target(targets, max_sequence_length=self._max_length,pad_token_id=self._pad_token_id)
        #entity_ids, entity_mask = self.pad_target(entities,max_sequence_length=self._max_length,pad_token_id=self._pad_token_id)
        target_ids, target_mask = self.pad_target(targets, max_sequence_length=64,pad_token_id=self._pad_token_id)
        entity_ids, entity_mask = self.pad_target(entities,max_sequence_length=64,pad_token_id=self._pad_token_id)

        output = input_ids, attention_mask, target_ids, target_mask, entity_ids, entity_mask

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


def get_dataloader_tag(num_workers,dataset, batch_size, max_len, pad_id, sampler):
    collate_fn = SmartBatchingCollateTag(
        max_length=max_len,
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


def get_dataloader_tag_pretrain(num_workers,dataset, batch_size, max_len, pad_id, sampler):
    collate_fn = SmartBatchingCollateTagPretrain(
        max_length=max_len,
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


def getpromptembedding(model, tokenizer, promptnumber, taskname):
    t5_embedding = model.model.get_input_embeddings()
    promptinitembedding = torch.FloatTensor(promptnumber, t5_embedding.weight.size(1))
    startindex = 0
    alllabel = ["summarization"]
    alllabel.append(taskname)
    for one in alllabel:
        encoderes = tokenizer.batch_encode_plus([one], padding=False, truncation=False, return_tensors="pt")
        touse = encoderes["input_ids"].squeeze()[:-1]
        embeddingres = t5_embedding(touse).clone().detach()
        if embeddingres.shape[0] > 1:
            embeddingres = torch.mean(embeddingres, 0, keepdim=True)
        promptinitembedding[startindex] = embeddingres
        startindex += 1
    fr = open('allnumber.pickle', 'rb')
    alltokens = pickle5.load(fr)
    sortedalltoken = sorted(alltokens.items(), key=lambda item: item[1], reverse=True)
    top5000 = []
    for one in sortedalltoken:
        if one[0] == 2:
            continue
        else:
            if len(top5000) < 5000:
                top5000.append(one)
            else:
                break
    vocab = tokenizer.get_vocab()
    randomtokennum = promptnumber - len(alllabel)
    touse = random.sample(top5000, randomtokennum)
    # print(touse)
    for one in touse:
        promptinitembedding[startindex] = t5_embedding.weight[one[0]].clone().detach()
        startindex += 1

    return promptinitembedding
