import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import sys
import argparse
import matplotlib
import pdb
import numpy as np
import time
import random
import time
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
from sklearn import metrics
from torch.utils import data
from collections import Counter
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)
from seqeval.metrics import classification_report,f1_score
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.grad_scaler import ShardedGradScaler
import pickle5 as pickle
from datasets import load_metric




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



