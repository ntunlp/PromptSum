from __future__ import absolute_import, division, print_function
import argparse
import gc
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch.nn.functional as F
from transformers import (AdamW, BertConfig, BertForTokenClassification, BertTokenizer,
                                  get_linear_schedule_with_warmup)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from seqeval.metrics import classification_report,f1_score

from util import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Ner(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None,attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask,head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cuda')
        #valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            #attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class NerCPU(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None,attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask,head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            #attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

def readfile(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        #print(splits)
        sentence.append(splits[0])
        label.append(splits[-1][:-1])
        #print(splits[-1][:-1])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []
    return data

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "docwithlabel_train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "docwithlabel_valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "conll_test.txt")), "test")

    def get_sum_examples(self, sum_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(sum_dir), "sum")

    def get_from_list(self, datalist):
        """See base class."""
        return self._create_examples(datalist, "datalist")

    def get_labels(self):
        return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]

    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            #print(i,sentence,label)
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list,1)}

    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        #textlist = example.text_a.split('\t')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0,1)
        label_mask.insert(0,1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
    return features

def dooneeval(model,eval_examples,label_list,args,tokenizer,device,max_seq_length=128, eval_batch_size=8):
    eval_features = convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                              all_lmask_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        valid_ids = valid_ids.to(device)
        label_ids = label_ids.to(device)
        l_mask = l_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids, attention_mask_label=l_mask)

        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()


        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == len(label_map):
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map[logits[i][j]])
    # for i in range(len(y_true)):
    #     print(y_true[i])
    #     print("&&&&&&&&&&&&&&&&&&&")
    #     print(y_pred[i])
    f1score = f1_score(y_true, y_pred)
    logger.info('----Validation Results Summary----')
    logger.info(f1score)
    return f1score


def getentitiesforonedata(newdata,bert_tagger_path,model,tokenizer,args):
    processors = {"ner": NerProcessor}
    task_name = 'ner'
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    # model = Ner.from_pretrained(bert_tagger_path)
    # tokenizer = BertTokenizer.from_pretrained(bert_tagger_path, do_lower_case=False)
    # model.to(args.device)

    sum_examples = processor.get_from_list(newdata)
    sum_features = convert_examples_to_features(sum_examples, label_list, 128, tokenizer)
    sum_input_ids = torch.tensor([f.input_ids for f in sum_features], dtype=torch.long)
    sum_input_mask = torch.tensor([f.input_mask for f in sum_features], dtype=torch.long)
    sum_segment_ids = torch.tensor([f.segment_ids for f in sum_features], dtype=torch.long)
    sum_label_ids = torch.tensor([f.label_id for f in sum_features], dtype=torch.long)
    sum_valid_ids = torch.tensor([f.valid_ids for f in sum_features], dtype=torch.long)
    sum_lmask_ids = torch.tensor([f.label_mask for f in sum_features], dtype=torch.long)
    sum_eval_data = TensorDataset(sum_input_ids, sum_input_mask, sum_segment_ids, sum_label_ids, sum_valid_ids,
                                  sum_lmask_ids)
    sum_eval_sampler = SequentialSampler(sum_eval_data)
    sum_eval_dataloader = DataLoader(sum_eval_data, sampler=sum_eval_sampler, batch_size=len(newdata))
    model.eval()

    sum_y_true = []
    sum_y_pred = []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    #for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(sum_eval_dataloader, desc="SumEvaluating"):
    for step, (input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask) in enumerate(sum_eval_dataloader):
        # input_ids = input_ids.to(args.device)
        # input_mask = input_mask.to(args.device)
        # segment_ids = segment_ids.to(args.device)
        # valid_ids = valid_ids.to(args.device)
        # label_ids = label_ids.to(args.device)
        # l_mask = l_mask.to(args.device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids, attention_mask_label=l_mask)

        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()

        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j].item() == len(label_map):
                    sum_y_true.append(temp_1)
                    sum_y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(label_map[label_ids[i][j].item()])
                    temp_2.append(label_map[logits[i][j].item()])
    torch.cuda.empty_cache()
    #del model, tokenizer
    gc.collect()

    for i in range(len(sum_y_pred)):
        assert len(sum_y_pred[i]) == len(newdata[i][0])

    allentitylist = []
    for i in range(len(sum_y_pred)):
        onelength = len(sum_y_pred[i])
        oneentitylist = []
        currententity = []
        for j in range(onelength):
            onepred = sum_y_pred[i][j]
            if onepred.find("B-") != -1:
                if currententity != []:
                    oneentitylist.append(' '.join(currententity))
                currententity = [newdata[i][0][j]]
            elif onepred.find("I-") != -1:
                currententity.append(newdata[i][0][j])
            else:
                continue
        if currententity != []:
            oneentitylist.append(' '.join(currententity))
        allentitylist.extend(oneentitylist)
    #print(allentitylist)
    return allentitylist


def get_predict_label_for_sum(args, doc_sum_path, sumpath):

    #####handle sumfile to fake conll format and use NER model to label it
    sumwithfakelabel = doc_sum_path + "sumwithfakelabel.txt"
    allsumwithfakelabeldata = getfilewithlabel(sumpath, sumwithfakelabel)

    processors = {"ner": NerProcessor}
    task_name = 'ner'
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    model = Ner.from_pretrained(args.pretrain_bert_path)
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_bert_path, do_lower_case=False)
    model.to(args.device)

    #####use sumwithfakelabel as test file to get the predicted label
    sum_examples = processor.get_sum_examples(sumwithfakelabel)
    #sum_features = convert_examples_to_features(sum_examples, label_list, 128, tokenizer)
    sum_features = convert_examples_to_features(sum_examples, label_list, 200, tokenizer) ####which length?
    logger.info("***** Running sumwithfakelabel *****")
    logger.info("  Num examples = %d", len(sum_examples))
    logger.info("  Batch size = %d", 32)
    sum_input_ids = torch.tensor([f.input_ids for f in sum_features], dtype=torch.long)
    sum_input_mask = torch.tensor([f.input_mask for f in sum_features], dtype=torch.long)
    sum_segment_ids = torch.tensor([f.segment_ids for f in sum_features], dtype=torch.long)
    sum_label_ids = torch.tensor([f.label_id for f in sum_features], dtype=torch.long)
    sum_valid_ids = torch.tensor([f.valid_ids for f in sum_features], dtype=torch.long)
    sum_lmask_ids = torch.tensor([f.label_mask for f in sum_features], dtype=torch.long)
    sum_eval_data = TensorDataset(sum_input_ids, sum_input_mask, sum_segment_ids, sum_label_ids, sum_valid_ids,
                                  sum_lmask_ids)
    sum_eval_sampler = SequentialSampler(sum_eval_data)
    sum_eval_dataloader = DataLoader(sum_eval_data, sampler=sum_eval_sampler, batch_size=32)
    model.eval()

    sum_y_true = []
    sum_y_pred = []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(sum_eval_dataloader,
                                                                                 desc="SumEvaluating"):
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        segment_ids = segment_ids.to(args.device)
        valid_ids = valid_ids.to(args.device)
        label_ids = label_ids.to(args.device)
        l_mask = l_mask.to(args.device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids, attention_mask_label=l_mask)

        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == len(label_map):
                    sum_y_true.append(temp_1)
                    sum_y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map[logits[i][j]])
    torch.cuda.empty_cache()
    del model, tokenizer
    gc.collect()

    for i in range(len(sum_y_pred)):
        #print(sum_y_pred[i])
        #print("&&&&&&&&&&&&&&&&&&&")
        #print(allsumwithfakelabeldata[i])
        if len(sum_y_pred[i]) != len(allsumwithfakelabeldata[i]):
            print("a error!")
            print(len(sum_y_pred[i]),len(allsumwithfakelabeldata[i]))
    return sum_y_pred, allsumwithfakelabeldata

# def get_doc_label(sum_y_pred,allsumwithfakelabeldata, docfile):
#
#     ####get all entities from sum_y_pred and allsumwithfakelabeldata. One problem: sum_y_pred might be wrong
#     allentitylist = []
#     alltypelist = []
#     #print(sum_y_pred)
#     for i in range(len(sum_y_pred)):
#         onelength = len(sum_y_pred[i])
#         oneentitylist = []
#         onetypelist = []
#         currententity = []
#         currenttype = []
#         for j in range(onelength):
#             onepred = sum_y_pred[i][j]
#             if onepred.find("B-") != -1:
#                 if currententity != []:
#                     oneentitylist.append(' '.join(currententity))
#                     onetypelist.append(' '.join(currenttype))
#                 currenttype = [onepred]
#                 currententity = [allsumwithfakelabeldata[i][j]]
#             elif onepred.find("I-") != -1:
#                 currenttype.append(onepred)
#                 currententity.append(allsumwithfakelabeldata[i][j])
#             else:
#                 continue
#         if currententity != []:
#             oneentitylist.append(' '.join(currententity))
#             onetypelist.append(' '.join(currenttype))
#         allentitylist.append(oneentitylist)
#         alltypelist.append(onetypelist)
#     #print(allentitylist)
#     ####combine doc and entities
#     alldocres, resfortrain = getdocandent(docfile, allentitylist, alltypelist)
#
#     ######handle resfortrain
#     #print(len(resfortrain))
#     #print("-------------------------------------")
#     allentityfortrain = []
#     for i in range(len(resfortrain)):
#         oneentityfortrain = []
#         onedata = resfortrain[i].split('\t')
#         onedoc = onedata[0]
#         oneent = onedata[1].split('!')
#         #print(oneent)
#         for j in range(len(oneent)):
#             enttouse = oneent[j]
#             if enttouse.lower() in onedoc.lower():
#                 oneentityfortrain.append(enttouse)
#         allentityfortrain.append([onedoc, oneentityfortrain])
#     # print(len(allentityfortrain))
#     # print("************************************")
#     # print(allentityfortrain)
#
#     ######get label for document
#     alldocandlabel = []
#     for i in range(len(alldocres)):
#         onedata = alldocres[i].split('\t')
#         onedoc = onedata[0]
#         length = len(onedoc.split(' '))
#         doclabel = ['O' for m in range(length)]
#         oneent = onedata[1].split('!')
#         onetype = onedata[2].split('?')
#         assert len(oneent) == len(onetype)
#         for j in range(len(oneent)):
#             enttouse = oneent[j]
#             typetouse = onetype[j]
#             if enttouse.lower() in onedoc.lower():
#                 ###add label
#                 typelist = typetouse.split(' ')
#                 allindex = getindex(enttouse, onedoc)
#                 for oneindex in allindex:
#                     for m in range(len(typelist)):
#                         doclabel[oneindex[m]] = typelist[m]
#             else:
#                 continue
#         assert len(onedoc.split(' ')) == len(doclabel)
#         #####onedoc doclabel
#         alldocandlabel.append([onedoc.split(' '), doclabel])
#
#     return alldocandlabel,allentityfortrain
#
# def get_train_valid(alldocandlabel, doc_sum_path, allentityfortrain):
#     docwithlabel_train = doc_sum_path + "docwithlabel_train.txt"
#     docwithlabel_vaid = doc_sum_path + "docwithlabel_valid.txt"
#     fout = open(docwithlabel_train, 'w')
#     fout_1 = open(docwithlabel_vaid, 'w')
#     fout.write("-DOCSTART- -X- -X- O\n")
#     fout_1.write("-DOCSTART- -X- -X- O\n")
#     fout.write("\n")
#     fout_1.write("\n")
#     for aa in range(len(alldocandlabel)):
#         onedata = alldocandlabel[aa]
#         datasize = len(onedata[0])
#         if aa % 2 == 0:
#             for i in range(datasize):
#                 fout.write(onedata[0][i] + " NNP B-NP " + onedata[1][i] + "\n")
#             fout.write("\n")
#         else:
#             for i in range(datasize):
#                 fout_1.write(onedata[0][i] + " NNP B-NP " + onedata[1][i] + "\n")
#             fout_1.write("\n")
#     fout.close()
#     fout_1.close()
#
#     ####save train ent
#     train_ent = doc_sum_path + "trainent.txt"
#     fe = open(train_ent, 'w')
#     for i in range(len(allentityfortrain)):
#         if allentityfortrain[i][1] != []:
#             fe.write(allentityfortrain[i][0] + "\t" + ' '.join(allentityfortrain[i][1]) + '\n')
#         else:
#             fe.write(allentityfortrain[i][0] + "\tnone\n")
#     fe.close()
#     return docwithlabel_train, docwithlabel_vaid


def get_doc_label(sum_y_pred,allsumwithfakelabeldata, docfile):

    ####get all entities from sum_y_pred and allsumwithfakelabeldata. One problem: sum_y_pred might be wrong
    allentitylist = []
    alltypelist = []
    #print(sum_y_pred)
    for i in range(len(sum_y_pred)):
        onelength = len(sum_y_pred[i])
        oneentitylist = []
        onetypelist = []
        currententity = []
        currenttype = []
        for j in range(onelength):
            onepred = sum_y_pred[i][j]
            if onepred.find("B-") != -1:
                if currententity != []:
                    oneentitylist.append(' '.join(currententity))
                    onetypelist.append(' '.join(currenttype))
                currenttype = [onepred]
                currententity = [allsumwithfakelabeldata[i][j]]
            elif onepred.find("I-") != -1:
                currenttype.append(onepred)
                currententity.append(allsumwithfakelabeldata[i][j])
            else:
                continue
        if currententity != []:
            oneentitylist.append(' '.join(currententity))
            onetypelist.append(' '.join(currenttype))
        allentitylist.append(oneentitylist)
        alltypelist.append(onetypelist)
    #print(allentitylist)
    ####combine doc and entities
    allrestrain, allresvalid, resfortrain = getdocandent(docfile,allentitylist,alltypelist)

    ######handle resfortrain
    #print(len(resfortrain))
    #print("-------------------------------------")
    allentityfortrain = []
    for i in range(len(resfortrain)):
        oneentityfortrain = []
        onedata = resfortrain[i].split('\t')
        onedoc = onedata[0]
        oneent = onedata[1].split('!')
        #print(oneent)
        for j in range(len(oneent)):
            enttouse = oneent[j]
            if enttouse.lower() in onedoc.lower():
                oneentityfortrain.append(enttouse)
        allentityfortrain.append([onedoc, oneentityfortrain])
    # print(len(allentityfortrain))
    # print("************************************")
    # print(allentityfortrain)

    ######get label for document
    alldocandlabeltrain = []
    for i in range(len(allrestrain)):
        onedata = allrestrain[i].split('\t')
        onedoc = onedata[0]
        length = len(onedoc.split(' '))
        doclabel = ['O' for m in range(length)]
        oneent = onedata[1].split('!')
        onetype = onedata[2].split('?')
        assert len(oneent) == len(onetype)
        allentinter = []
        for j in range(len(oneent)):
            enttouse = oneent[j]
            typetouse = onetype[j]
            if enttouse.lower() in onedoc.lower():
                allentinter.append(enttouse)
                ###add label
                typelist = typetouse.split(' ')
                allindex = getindex(enttouse, onedoc)
                for oneindex in allindex:
                    for m in range(len(typelist)):
                        doclabel[oneindex[m]] = typelist[m]
            else:
                continue
        assert len(onedoc.split(' ')) == len(doclabel)
        #####onedoc doclabel
        alldocandlabeltrain.append([onedoc.split(' '), doclabel])

    alldocandlabelvalid = []
    for i in range(len(allresvalid)):
        onedata = allresvalid[i].split('\t')
        onedoc = onedata[0]
        length = len(onedoc.split(' '))
        doclabel = ['O' for m in range(length)]
        oneent = onedata[1].split('!')
        onetype = onedata[2].split('?')
        assert len(oneent) == len(onetype)
        allentinter = []
        for j in range(len(oneent)):
            enttouse = oneent[j]
            typetouse = onetype[j]
            if enttouse.lower() in onedoc.lower():
                allentinter.append(enttouse)
                ###add label
                typelist = typetouse.split(' ')
                allindex = getindex(enttouse, onedoc)
                for oneindex in allindex:
                    for m in range(len(typelist)):
                        doclabel[oneindex[m]] = typelist[m]
            else:
                continue
        assert len(onedoc.split(' ')) == len(doclabel)
        alldocandlabelvalid.append([onedoc.split(' '), doclabel])

    #print(len(alldocandlabel))
    return alldocandlabeltrain,alldocandlabelvalid,allentityfortrain

def get_train_valid(alldocandlabeltrain, alldocandlabelvalid, doc_sum_path, allentityfortrain):
    docwithlabel_train = doc_sum_path + "docwithlabel_train.txt"
    docwithlabel_vaid = doc_sum_path + "docwithlabel_valid.txt"
    fout = open(docwithlabel_train, 'w')
    fout_1 = open(docwithlabel_vaid, 'w')
    fout.write("-DOCSTART- -X- -X- O\n")
    fout_1.write("-DOCSTART- -X- -X- O\n")
    fout.write("\n")
    fout_1.write("\n")

    for aa in range(len(alldocandlabeltrain)):
        onedata = alldocandlabeltrain[aa]
        datasize = len(onedata[0])
        for i in range(datasize):
            fout.write(onedata[0][i] + " NNP B-NP " + onedata[1][i] + "\n")
        fout.write("\n")
    fout.close()

    for aa in range(len(alldocandlabelvalid)):
        onedata = alldocandlabelvalid[aa]
        datasize = len(onedata[0])
        for i in range(datasize):
            fout_1.write(onedata[0][i] + " NNP B-NP " + onedata[1][i] + "\n")
        fout_1.write("\n")
    fout_1.close()

    ####save train ent
    train_ent = doc_sum_path + "trainent.txt"
    fe = open(train_ent, 'w')
    for i in range(len(allentityfortrain)):
        if allentityfortrain[i][1] != []:
            fe.write(allentityfortrain[i][0] + "\t" + ' '.join(allentityfortrain[i][1]) + '\n')
        else:
            fe.write(allentityfortrain[i][0] + "\tnone\n")
    fe.close()
    return docwithlabel_train, docwithlabel_vaid


def finetune_model(trainfile, validfile, args):
    processors = {"ner": NerProcessor}
    task_name = 'ner'
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1
    print(trainfile, validfile)

    ###train
    gradient_accumulation_steps = 1
    train_batch_size = 4
    eval_batch_size = 8
    num_train_epochs = 30
    warmup_proportion = 0.1
    learning_rate = 5e-5
    adam_epsilon = 1e-8
    weight_decay = 0.01
    max_grad_norm = 1.0
    max_seq_length = 128

    train_batch_size = train_batch_size // gradient_accumulation_steps

    #####the path of tuned model
    pos = trainfile.find("docwithlabel_train")
    foldername = trainfile[0:pos]
    print(foldername)
    output_dir = foldername + "tagger"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_bert_path, do_lower_case=False)

    train_examples = processor.get_train_examples(foldername)
    num_train_optimization_steps = int(
        len(train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Prepare model

    model = Ner.from_pretrained(args.pretrain_bert_path)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(warmup_proportion * num_train_optimization_steps)

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    global_step = 0

    train_features = convert_examples_to_features(
        train_examples, label_list, max_seq_length, tokenizer)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                               all_lmask_ids)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    model.train()
    bestevalscore = -100
    for i in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                if scheduler != None:
                    scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            eval_examples = processor.get_dev_examples(foldername)
            thisevalscore = dooneeval(model, eval_examples, label_list, args, tokenizer, args.device, max_seq_length, eval_batch_size)
            if thisevalscore > bestevalscore:
                logger.info('save best model')
                bestevalscore = thisevalscore
                # save
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
    ###kill

    torch.cuda.empty_cache()
    del model, tokenizer
    gc.collect()

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default="./sumdata/cnndm/0_42/",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default="bert-base-cased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default="ner",
                        type=str,
                        help="The name of the task to train.")

    parser.add_argument("--load_dir",
                        default="out_base",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--output_dir",
                        default="output",
                        type=str,
                        help="The output directory of model after training on sum.")

    ## Other parameters
    parser.add_argument('--cache_dir', type=str, default="/export/home/cache")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_false',
                        #action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_false',
                        help="Whether to run eval or not.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--test_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for test.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--sumdata', type=str, default='./sumdata/cnndm/0_42', help="The path of original sum data.")

    args = parser.parse_args()

    processors = {"ner":NerProcessor}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    #### load from load_dir
    model = Ner.from_pretrained(args.load_dir)
    tokenizer = BertTokenizer.from_pretrained(args.load_dir, do_lower_case=args.do_lower_case)
    model.to(device)

    #####load original sum data and process it: seperate it to document + summary.
    sumdatapath = args.sumdata  ####default: ./sumdata/cnndm/0_42   get_doc_and_sum function use 'train.txt' and 'valid.txt' under this folder
    usetrain = True
    usevalid = True
    docfile, sumfile = get_doc_and_sum(sumdatapath, usetrain, usevalid)

    #####handle sumfile to fake conll format and use NER model to label it
    sumwithfakelabel = sumdatapath + "/sumwithfakelabel.txt"
    allsumwithfakelabeldata = getfilewithlabel(sumfile, sumwithfakelabel)

    #####use sumwithfakelabel as test file to get the predicted label
    sum_examples = processor.get_sum_examples(sumwithfakelabel)
    sum_features = convert_examples_to_features(sum_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running sumwithfakelabel *****")
    logger.info("  Num examples = %d", len(sum_examples))
    logger.info("  Batch size = %d", args.test_batch_size)
    sum_input_ids = torch.tensor([f.input_ids for f in sum_features], dtype=torch.long)
    sum_input_mask = torch.tensor([f.input_mask for f in sum_features], dtype=torch.long)
    sum_segment_ids = torch.tensor([f.segment_ids for f in sum_features], dtype=torch.long)
    sum_label_ids = torch.tensor([f.label_id for f in sum_features], dtype=torch.long)
    sum_valid_ids = torch.tensor([f.valid_ids for f in sum_features], dtype=torch.long)
    sum_lmask_ids = torch.tensor([f.label_mask for f in sum_features], dtype=torch.long)
    sum_eval_data = TensorDataset(sum_input_ids, sum_input_mask, sum_segment_ids, sum_label_ids, sum_valid_ids,
                              sum_lmask_ids)
    sum_eval_sampler = SequentialSampler(sum_eval_data)
    sum_eval_dataloader = DataLoader(sum_eval_data, sampler=sum_eval_sampler, batch_size=args.test_batch_size)
    model.eval()

    sum_y_true = []
    sum_y_pred = []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(sum_eval_dataloader, desc="SumEvaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        valid_ids = valid_ids.to(device)
        label_ids = label_ids.to(device)
        l_mask = l_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids, attention_mask_label=l_mask)

        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()

        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == len(label_map):
                    sum_y_true.append(temp_1)
                    sum_y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map[logits[i][j]])
    torch.cuda.empty_cache()
    del model, tokenizer
    gc.collect()

    for i in range(len(sum_y_pred)):
        if len(sum_y_pred[i]) != len(allsumwithfakelabeldata[i]):
            print("a error!")

    ####get all entities from sum_y_pred and allsumwithfakelabeldata. One problem: sum_y_pred might be wrong
    allentitylist = []
    alltypelist = []
    for i in range(len(sum_y_pred)):
        onelength = len(sum_y_pred[i])
        oneentitylist = []
        onetypelist = []
        currententity = []
        currenttype = []
        for j in range(onelength):
            onepred = sum_y_pred[i][j]
            if onepred.find("B-") != -1:
                if currententity != []:
                    oneentitylist.append(' '.join(currententity))
                    onetypelist.append(' '.join(currenttype))
                currenttype = [onepred]
                currententity = [allsumwithfakelabeldata[i][j]]
            elif onepred.find("I-") != -1:
                currenttype.append(onepred)
                currententity.append(allsumwithfakelabeldata[i][j])
            else:
                continue
        if currententity != []:
            oneentitylist.append(' '.join(currententity))
            onetypelist.append(' '.join(currenttype))
        allentitylist.append(oneentitylist)
        alltypelist.append(onetypelist)

    ####combine doc and entities
    alldocres, resfortrain = getdocandent(docfile,allentitylist,alltypelist)


    ######get label for document
    alldocandlabel = []
    for i in range(len(alldocres)):
        onedata = alldocres[i].split('\t')
        onedoc = onedata[0]
        length = len(onedoc.split(' '))
        doclabel = ['O' for m in range(length)]
        oneent = onedata[1].split('!')
        onetype = onedata[2].split('?')
        assert len(oneent) == len(onetype)
        for j in range(len(oneent)):
            enttouse = oneent[j]
            typetouse = onetype[j]
            if enttouse.lower() in onedoc.lower():
                ###add label
                typelist = typetouse.split(' ')
                allindex = getindex(enttouse, onedoc)
                for oneindex in allindex:
                    for m in range(len(typelist)):
                        doclabel[oneindex[m]] = typelist[m]
            else:
                continue
        assert len(onedoc.split(' ')) == len(doclabel)
        #####onedoc doclabel
        alldocandlabel.append([onedoc.split(' '), doclabel])

    print(len(alldocandlabel))

    docwithlabel_train = sumdatapath + "/docwithlabel_train.txt"
    docwithlabel_vaid = sumdatapath + "/docwithlabel_valid.txt"
    fout = open(docwithlabel_train, 'w')
    fout_1 = open(docwithlabel_vaid, 'w')
    fout.write("-DOCSTART- -X- -X- O\n")
    fout_1.write("-DOCSTART- -X- -X- O\n")
    fout.write("\n")
    fout_1.write("\n")
    for aa in range(len(alldocandlabel)):
        onedata = alldocandlabel[aa]
        datasize = len(onedata[0])
        if aa % 2 == 0:
            for i in range(datasize):
                fout.write(onedata[0][i] + " NNP B-NP " + onedata[1][i] + "\n")
            fout.write("\n")
        else:
            for i in range(datasize):
                fout_1.write(onedata[0][i] + " NNP B-NP " + onedata[1][i] + "\n")
            fout_1.write("\n")
    fout.close()
    fout_1.close()

    ###train
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.load_dir, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = 0
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Prepare model

    model = Ner.from_pretrained(args.load_dir)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    global_step = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        bestevalscore = -100
        for i in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids, valid_ids,l_mask)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    if scheduler != None:
                        scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
            if i >= 0 and args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                eval_examples = processor.get_dev_examples(args.data_dir)
                thisevalscore = dooneeval(model,eval_examples,label_list,args,tokenizer,device)
                if thisevalscore > bestevalscore:
                    logger.info('save best model')
                    bestevalscore = thisevalscore
                    #save
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    model_to_save.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
