import os
import sys
sys.path.append("./tagger/")
import torch
import numpy as np
import random
from tagger.TrainTaggerforSum import *
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)

def seed_everything(args):
    random.seed(args.seed)
    os.environ['PYTHONASSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_model(modeltoeval, args, steps):
    if isinstance(modeltoeval, torch.nn.parallel.DistributedDataParallel):
        model = modeltoeval.module
    else:
        model = modeltoeval
    model.eval()
    if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
    if not os.path.exists(args.save_path + "/" + args.save_dir):
        os.mkdir(args.save_path + "/" + args.save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    if args.model == 'T5Prompt':
        ckpt = {
            "prompt_length": model_to_save.prompt_length,
            "prompt_embedding": model_to_save.prompt_embedding
        }
    elif args.model == 'T5MixPrompt':
        ckpt = {
            "prompt_dict": model_to_save.prompt_dict,
            "prompt_fix_dict": model_to_save.prompt_fix_dict
        }
    elif args.model == 'T5Finetune':
        ckpt = {
            't5-base': model_to_save.model.state_dict(),
        }
    print("about to save")
    torch.save(ckpt, os.path.join(args.save_path + "/" + args.save_dir, "ckptofT5_"+str(steps)))
    print("ckpt saved")

# def get_train_valid_data(args, sumpath, docpath, doc_sum_path):
#
#     ####get predict label of summarization
#     sum_y_pred, allsumwithfakelabeldata = get_predict_label_for_sum(args, doc_sum_path, sumpath)
#
#     ####get label for document
#     alldocandlabel, allentityfortrain = get_doc_label(sum_y_pred, allsumwithfakelabeldata, docpath)
#
#     ####split to train and valid
#     docwithlabel_train, docwithlabel_vaid = get_train_valid(alldocandlabel, doc_sum_path, allentityfortrain)
#
#     return docwithlabel_train, docwithlabel_vaid

def get_train_valid_data(args, sumpath, docpath, doc_sum_path):

    ####get predict label of summarization
    sum_y_pred, allsumwithfakelabeldata = get_predict_label_for_sum(args, doc_sum_path, sumpath)

    ####get label for document
    alldocandlabeltrain, alldocandlabelvalid,allentityfortrain = get_doc_label(sum_y_pred,allsumwithfakelabeldata, docpath)

    ####split to train and valid
    docwithlabel_train, docwithlabel_vaid = get_train_valid(alldocandlabeltrain, alldocandlabelvalid, doc_sum_path, allentityfortrain)

    return docwithlabel_train, docwithlabel_vaid
def train_tagger_for_one_seed(trainfile, validfile, args):
    finetune_model(trainfile, validfile, args)