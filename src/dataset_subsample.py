import os
import sys
from collections import defaultdict
import random 
import re

def get_line_labels(line):
    _, labels = line.strip().split('\t')
    labels = labels.split(';')
    label_list = []
    for label in labels:
        if not label: # pass empty str
            continue
        if '!' in label:
            label_list.append(re.search('.*!(.*)', label)[1])
        else: # end
            assert label == 'end'
            label_list.append(label)
    return label_list 

def subsample(source, target, n, SEED):
    '''
    Args:
        source: file path to source file, data line by line
        target: file path to target file
        n[int]: number of examples kept for each label class 
    '''
    random.seed(SEED)
    all_data = []
    idx_chosen = set()
    labels_set = set()
    labels_count = {}
    with open(source, 'r') as rf:
        all_lines = rf.readlines()

    # get label set
    for l_idx, line in enumerate(all_lines):
        labels = get_line_labels(line)
        labels_set.update(labels)
    # init labels_count
    labels_count = {la:0 for la in labels_set}
    
    # add new examples, until all labels have at least n examples
    while any([la_cnt < n for la_cnt in labels_count.values()]):
        # randomly sample one line
        l_idx = random.randint(0, len(all_lines)-1)
        if l_idx not in idx_chosen:
            l_labels = get_line_labels(all_lines[l_idx])
            if all([labels_count[la] < n for la in l_labels]):
                idx_chosen.add(l_idx)
                for l_label in l_labels:
                    labels_count[l_label] += 1

    # write new data to file
    idx_chosen = list(idx_chosen)
    random.shuffle(idx_chosen)
    with open(target, 'w') as wf:
        for idx in idx_chosen:
            wf.write(all_lines[idx])
    print(f"Subsampling {source} to {target}, with labels: {labels_count}")

def subsample_cl(source, target, n, SEED, cl_stages={1:set([' person ', 'end']), 2:set([' location ']), 3:set([' org ']), 4:set([' mix '])}):
    '''
    Args:
        source: file path to source file, data line by line
        target: file path to target file
        n[int]: number of examples kept for each label class 
        cl_stages: dict of stage_num: label_list, to indicate new labels for each stage during continue learning
    '''
    random.seed(SEED)
    all_data = []
    all_idx_chosen = set()
    labels_set = set()
    labels_count = {}
    with open(source, 'r') as rf:
        all_lines = rf.readlines()

    # get label set
    for l_idx, line in enumerate(all_lines):
        labels = get_line_labels(line)
        labels_set.update(labels)
    
    seen_labels = set(['dummy']) # 
    for stage_num,new_labels in cl_stages.items():
        idx_chosen = set()
        # init labels_count
        labels_count = {la:0 for la in new_labels}
        # update seen labels
        seen_labels = seen_labels.union(new_labels)
        # add new examples, until every new label has at least n examples
        while any([la_cnt < n for la_cnt in labels_count.values()]):
            # randomly sample one line
            l_idx = random.randint(0, len(all_lines)-1)
            if l_idx not in idx_chosen:
                l_labels = get_line_labels(all_lines[l_idx])
                if set(l_labels).issubset(seen_labels) and any([la in new_labels for la in l_labels]) and all([labels_count[la] < n for la in l_labels if la in labels_count]):
                    idx_chosen.add(l_idx)
                    all_idx_chosen.add(l_idx)
                    for l_label in set(l_labels):
                        if l_label in labels_count:
                            labels_count[l_label] += 1

        # write new data to file
        idx_chosen = list(idx_chosen)
        random.shuffle(idx_chosen)
        this_target = target+'_'+str(stage_num)+'.txt'
        with open(this_target, 'w') as wf:
            for idx in idx_chosen:
                wf.write(all_lines[idx])
        print(f"Subsampling {source} to {this_target}, with labels: {labels_count}")

if __name__ == '__main__':
    # subsample('./data/data_conll/newvalid.txt', './data/data_conll/tiny_valid.txt', 16, SEED=30)
    # subsample('./data/data_conll/newtest.txt', './data/data_conll/tiny_test.txt', 16, SEED=30)
    # subsample('./data/data_conll/newtrain.txt', './data/data_conll/tiny_train.txt', 16, SEED=30)
    subsample_cl('./data/data_conll/newtrain.txt', './data/data_conll/fewshot_cl/train', 16, SEED=30)
    subsample_cl('./data/data_conll/newvalid.txt', './data/data_conll/fewshot_cl/valid', 16, SEED=30)
    

    
