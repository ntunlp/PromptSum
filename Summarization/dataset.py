import sys
sys.path.append("../..")

import spacy
import torch
import datasets
import os
import numpy as np

from torch.utils.data import Sampler, Dataset, DataLoader
from rouge_score import rouge_scorer


class T5SummarizationDataset(Dataset):
    def __init__(self, filename, split, maxlen, tokenizer, newtgentasktokens, answertoken, args):
        super(T5SummarizationDataset, self).__init__()
        self.filename = filename
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.gentasktoken = newtgentasktokens
        self.answertoken = answertoken
        self.args = args
        
        self.data = []
        self.data = self.getalldata(self.filename)
        self.num_entries = len(self.data)

        self.split = split
        self.tagger = None
        self.tagtokenizer = None
        self.bert_tagger_path = ""
        self.allent = {}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        if args.use_bert_tagger:
            ####train valid test
            if self.split.startswith("train"):
                ####load entity file for training data
                entpath = f'{save_path}data_for_bert_{seed}/trainent.txt'
                fe = open(entpath,'r')
                while True:
                    oneline = fe.readline().strip()
                    if not oneline:
                        break
                    content = oneline.split("\t")
                    if len(content) != 2:
                        print("train data entity error!!!!")
                        continue
                    doc = content[0]

                    templist = content[1].split(' ')

                    # try:
                    #     matches = list(datefinder.find_dates(doc))
                    # except:
                    #     print("one except")
                    # print(len(matches))
                    # templist.extend([str(aa) for aa in matches])

                    # allnum = [s for s in doc.split(' ') if s.isdigit()]
                    # allnum = list(set(allnum))
                    # #print(allnum)
                    # templist.extend(allnum)

                    entlist = ','.join(templist)
                    #print(entlist)
                    self.allent[doc] = entlist
                fe.close()
            else:
                print(f'We are in {self.split} mode. We should use a bert tagger to predict entities.')
                self.bert_tagger_path = f'{save_path}data_for_bert_{seed}/tagger/'

                ####use tuned tagger
                self.tagger = NerCPU.from_pretrained(self.bert_tagger_path)
                self.tagtokenizer = BertTokenizer.from_pretrained(self.bert_tagger_path, do_lower_case=False)

                ####use tagger pretrained on conll
                # self.tagger = NerCPU.from_pretrained(self.args.pretrain_bert_path)
                # self.tagtokenizer = BertTokenizer.from_pretrained(self.args.pretrain_bert_path, do_lower_case=False)

        # counterfactual training
        self.counterfactual_removal = args.counterfactual_removal
        if self.counterfactual_removal:
            self.counterfactual_remove()
            print("# After augmenting, Data points in this split: {}".format(len(self.data[self.args.text_key])))

    def getalldata(self,filename):
        f = open(filename,'r')
        alldata = []
        while True:
            oneline = f.readline().strip()
            if not oneline:
                break
            linelist = oneline.split("\t")
            onedata = []
            onedata.append(linelist[0])
            onedata.append(linelist[1])
            alldata.append(onedata)
        f.close()
        
        return alldata

    def __getitem__(self, idx):
        inputdata = self.data[idx][0]
        targetdata = self.data[idx][1]

        # guidance
        input_guidance = "None"
        # 1st option: based on entities
        if self.args.guidance_type == "ents":
            if not self.args.use_bert_tagger:
                if self.args.guidance_mode == 'oracle':
                    ents_x = self.spacy_nlp(inputdata).ents
                    ents_x = [ent.text for ent in ents_x]
                    ents_y = self.spacy_nlp(targetdata).ents
                    ents_y = [ent.text for ent in ents_y]
                    ents_intersection = [ent for ent in ents_x if ent in ents_y]
                    ents_intersection = list(dict.fromkeys(ents_intersection)) # remove duplicates, while keeping order
                    input_guidance = ' '.join(ents_intersection)
                else:
                    ents = self.spacy_nlp(inputdata).ents
                    ents = [ent.text for ent in ents]
                    input_guidance = ' '.join(ents) # can decide which delimiter works the best, just pick comma first
            else:
                ####for train
                if self.split.startswith("train"):
                    tempdata = re.sub(' +', ' ', inputdata)
                    ####search inputdata in the dic self.allent
                    if tempdata in self.allent.keys():
                        input_guidance = self.allent[tempdata]
                    else:
                        print("we can not find inputdata in the dictionary!! There should be some errors!")
                else:
                    ####we use the tuned entity model to predict entities
                    tempdata = re.sub(' +', ' ', inputdata)
                    templist = tempdata.split(' ')
                    num = 100
                    newlist = []
                    for j in range(0, len(templist), num):
                        newlist.append(templist[j:j + num])
                    ####handle newlist
                    newdata = []
                    for j in range(len(newlist)):
                        onedata = newlist[j]
                        onelabel = ['O' for oned in onedata]
                        newdata.append((onedata, onelabel))
                    allentitylist = getentitiesforonedata(newdata,self.bert_tagger_path,self.tagger,self.tagtokenizer, self.args)

                    # matches = datefinder.find_dates(tempdata)
                    # allentitylist.extend([str(aa) for aa in matches])

                    # allnum = [s for s in tempdata.split(' ') if s.isdigit()]
                    # allnum = list(set(allnum))
                    # allentitylist.extend(allnum)

                    input_guidance = ','.join(list(set(allentitylist)))
                    #print(input_guidance)
                    if input_guidance == []:
                        print("empty!")
                        ents = self.spacy_nlp(inputdata).ents
                        ents = [ent.text for ent in ents]
                        if self.args.filter_ents_freq:
                            ents = [x for x in ents if
                                    x in self.ents_freq.keys() and self.ents_freq[x] >= self.args.min_ents_freq]
                        input_guidance = ','.join(ents)  # can decide which delimiter works the best, just pick comma first
                        #print(input_guidance)

        # 2nd option: based on salient sentences
        elif self.args.guidance_type == "sents":
            salient_sents = build_salient_sents(inputdata, targetdata, self.rouge_scorer, self.args)
            input_guidance = ' '.join(salient_sents)  # can decide which delimiter works the best, just pick comma first

        inputres = self.tokenizer.batch_encode_plus([inputdata], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        targetres = self.tokenizer.batch_encode_plus([targetdata], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        input_ents_res = self.tokenizer.batch_encode_plus([input_guidance], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")

        return inputres["input_ids"].squeeze(), targetres["input_ids"].squeeze(), input_ents_res['input_ids'].squeeze()

    def __len__(self):
        
        return self.num_entries

    def counterfactual_remove(self):
        '''
        Function to add counterfactually removed instances to data
        input:
            data: has fields article, highlights, id
        '''
        inputdata = self.data[self.args.text_key]
        targetdata = self.data[self.args.summary_key]
        new_inputdata = inputdata
        new_targetdata = targetdata
        removed_ents = [None]*len(inputdata)
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
                    if len(removed)>0:
                        # construct counterfactual example
                        new_targetdata.append(' '.join(sents).replace(sent, ''))
                        new_inputdata.append(inputdata[i])
                        removed_ents.append(removed)
        # change self.data
        self.data = datasets.Dataset.from_dict({self.args.text_key: new_inputdata, self.args.summary_key: new_targetdata, 'removed_ents': removed_ents})


class SmartBatchingCollate:
    def __init__(self, max_length, max_guidance_length, pad_token_id):
        self._max_length = max_length
        self._max_guidance_length = max_guidance_length
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
            max_sequence_length=self._max_length, 
            pad_token_id=self._pad_token_id
        )

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
        
        return padded_sequences,attention_masks

    def pad_sequence(self, sequence_batch, max_sequence_length, pad_token_id):
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
        train_dataset = T5SummarizationDataset(train_file_name, "train", args.max_length, tokenizer, allgentasktokens, answertoken, args)
        valid_dataset = T5SummarizationDataset(valid_file_name, "valid", args.max_length, tokenizer, allgentasktokens, answertoken, args)
        datasets.append((train_dataset, valid_dataset))
    
    return datasets

def convert_data_to_txt(train_data, new_train_path, args):
    all_train_texts, all_train_summaries = [], []
    for idx in range(len(train_data)):
        text = train_data[idx][args.text_key]
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
