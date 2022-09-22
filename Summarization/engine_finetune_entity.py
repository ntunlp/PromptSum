import pickle
import argparse
import gc
import spacy
import time
import logging

gc.enable()

from datasets import load_metric
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, PegasusConfig
from torch.cuda.amp import autocast as autocast
from torch.utils import data
from torch.utils.data import (
    SequentialSampler, RandomSampler
)
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.grad_scaler import ShardedGradScaler

from utils import *
from dataset_pretrain import *
from dataset_finetune_entity import *
from models_summarization.model_soft import *
from model_finetune_entity import *
from engine_pretrain import *



def train_tagger_for_all_seeds(alltrainfile, allvalidfile, alltestfile, args):
    all_f1s, all_meanRs = [], []
    for i in range(len(alltrainfile)):
        result_dict = train_tagger_for_one_seed(alltrainfile[i], allvalidfile[i], alltestfile[i], args)
        f1 = result_dict["best_val_F1"]
        meanR = result_dict["best_val_meanR"]
        all_f1s.append(f1)
        all_meanRs.append(meanR)
    f1 = np.mean(all_f1s)
    clean_f1s = ["{:.4f}".format(x) for x in all_f1s]
    print("Mean F1: {:.4f} (over all seeds: {})".format(f1, clean_f1s))
    meanR = np.mean(all_meanRs)
    clean_meanRs = ["{:.4f}".format(x) for x in all_meanRs]
    print("Mean mean ROUGE: {:.4f} (over all seeds: {})".format(meanR, clean_meanRs))


def train_tagger_for_one_seed(trainfile, validfile, testfile, args):
    result_dict = finetune_model_tagger(trainfile, validfile, testfile, args)

    return result_dict


def finetune_model_tagger(trainfile, validfile, testfile, args):
    print("Fine-tuning entity tagger...")
    print(trainfile, validfile, testfile)

    ###train
    gradient_accumulation_steps = args.gradient_accumulation_steps_entity
    train_batch_size = args.batch_size_per_gpu_entity
    eval_batch_size = args.valid_size_per_gpu_entity
    num_train_epochs = args.max_epoch_entity  ### epochs for training tagger
    learning_rate = args.lr_entity
    weight_decay = args.weight_decay_entity
    max_seq_length = args.max_length
    num_workers = args.num_workers_entity
    max_grad_norm = args.max_grad_norm_entity
    log_step = args.log_step_finetune
    model_name = args.model_name

    if 't5' in args.model_name:
        basemodel = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir = args.cache_path)
        tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir = args.cache_path)
    elif 'pegasus' in args.model_name:
        basemodel = PegasusForConditionalGeneration.from_pretrained(model_name, max_position_embeddings = args.max_position_embeddings, cache_dir = args.cache_path)
        tokenizer = PegasusTokenizer.from_pretrained(model_name, cache_dir = args.cache_path)
    else:
        raise Exception('Model not implemented yet')
    model = ModelforFinetuneEntity(basemodel, tokenizer, args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("The model has {} trainable parameters".format(n_params))

    ##### load from conll ckpt, from pre-training ckpt, or simply initializing?
    if args.use_pretrain_ckpt:
        print("Loading the pre-trained NER model!")

        # model weights
        ckpt = torch.load(args.pretrain_ckpt, map_location="cuda:0")
        dic = {}
        for x in ckpt.keys():
            if (args.dataset == "billsum") and ("embed_positions" in x):
                continue
            if not (x in ["module.promptnumber", "module.promptembedding", "module.promptnumberforsum", "module.promptembeddingforsum"]):
               dic[x[7:]] = ckpt[x]
        if (args.max_position_embeddings > 1024):
            dic["model.model.encoder.embed_positions.weight"] = basemodel.state_dict()["model.encoder.embed_positions.weight"]
            dic["model.model.decoder.embed_positions.weight"] = basemodel.state_dict()["model.decoder.embed_positions.weight"]
        model.load_state_dict(dic)

        # just prompt
        ckpt = torch.load(args.pretrain_prompt_ckpt)
        model.promptnumber = ckpt["promptnumber"]
        model.promptembedding = nn.parameter.Parameter(ckpt["promptembedding"])
    else:
        ifuseconll = True
        if ifuseconll:
            print("Loading the the CONLL NER model!")
            allckpt = torch.load("./prompt_tuning_ckpt_conll/bestckpt")
            model.promptnumber = allckpt["promptnumber"]
            model.promptembedding = allckpt["promptembedding"]
        else:
            print("Initializing from scratch!")
            promptnumber = 300
            taskname = "name entity recognition"
            promptembedding = getpromptembedding(model, tokenizer, promptnumber, taskname)
            print("prompt", promptembedding.shape)
            model.set_prompt_embedding(promptnumber, promptembedding)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("The model has {} trainable parameters".format(n_params))
    model.to(args.device)

    train_dataset = T5DatasetPretrainConll(trainfile, max_seq_length, tokenizer)
    valid_dataset = T5DatasetPretrainConll(validfile, max_seq_length, tokenizer)
    test_dataset = T5DatasetPretrainConll(testfile, max_seq_length, tokenizer)

    if args.local_rank != -1:
        torch.distributed.barrier()

    if args.few_shot == "1":
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)
    test_sampler = SequentialSampler(test_dataset)

    train_dataloader = get_dataloader_tag(num_workers, train_dataset, train_batch_size, max_seq_length,
                                          train_dataset.tokenizer.pad_token_id, train_sampler)
    valid_dataloader = get_dataloader_tag(num_workers, valid_dataset, eval_batch_size, max_seq_length,
                                          valid_dataset.tokenizer.pad_token_id, valid_sampler)
    test_dataloader = get_dataloader_tag(num_workers, test_dataset, eval_batch_size, max_seq_length,
                                          test_dataset.tokenizer.pad_token_id, test_sampler)

    print(len(train_dataloader), len(valid_dataloader), len(test_dataloader))
    
    #####the path of tuned model
    pos = trainfile.find("docwithlabel_train")
    foldername = trainfile[0:pos]
    print(foldername)
    seedname = foldername.split("/")[-3]
    print(seedname)

    taggerfolder = "tagger_ckpt/"
    if not os.path.exists(taggerfolder):
        os.makedirs(taggerfolder)

    output_dir = taggerfolder + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_dir = output_dir + "/" + str(args.few_shot)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_dir = output_dir + "/" + seedname
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)

    base_optimizer_arguments = {
        "lr": learning_rate,
        "clip_threshold": max_grad_norm,
        "decay_rate": -0.8,
        "weight_decay": weight_decay,
        "scale_parameter": False,
        "relative_step": False
    }
    if args.optimizer_entity == "adafactor":
        optimizer = Adafactor(params=filter(lambda p: p.requires_grad, model.parameters()), **base_optimizer_arguments)

    model.train()

    startepoch = 0
    Best_F1, Best_val_meanR = -1.0, -1.0

    logger.info("Begin train...")

    result_dict = {
        'epoch': [],
        'val_F1': [],
        'best_val_F1': Best_F1,
        'val_r1': [],
        'val_r2': [],
        'val_rl': [],
        'best_val_meanR': Best_val_meanR
    }

    global_step = 0
    
    if args.eval_epoch_0:
        print("Evaluating (Epoch 0)...")
        dooneeval(model, valid_dataloader, result_dict, 0, output_dir, args, save_model=args.save_model)

    for i in range(startepoch, startepoch + num_train_epochs):
        logger.info(i)
        model.train()
        result_dict['epoch'] = i
        allloss = []
        for step, batch in enumerate(train_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
            loss = model(inputs)
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            allloss.append(loss.item())

            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [0, -1] and global_step % log_step == 0:
                    logger.info("step: %d,  loss: %.6f" % (global_step, np.average(allloss)))

                if args.local_rank in [0, -1] and global_step % args.eval_step_entity == 0:
                    print("only eval after every epoch")
                    model.train()

        logger.info("finish one epoch")
        if args.local_rank in [0, -1]:
            dooneeval(model, valid_dataloader, result_dict, i, output_dir, args, save_model=args.save_model)

    # test inference
    if args.full_testset:
        print("Test evaluation...")
        test_result_dict = {
            'epoch': [],
            'val_F1': [],
            'best_val_F1': Best_F1,
            'val_r1': [],
            'val_r2': [],
            'val_rl': [],
            'best_val_meanR': Best_val_meanR
        }
        if args.tune_weights:
            onepath = os.path.join(output_dir, "bestckpt_full_weights")
            if args.use_pretrain_ckpt:
                onepath += "_from_pretrained"
            oneckpt = torch.load(onepath)
            model.load_state_dict(oneckpt)
            print("loaded all model weights")
        else:
            onepath = os.path.join(output_dir, "bestckpt_prompt")
            if args.use_pretrain_ckpt:
                onepath += "_from_pretrained"
            oneckpt = torch.load(onepath)
            model.promptnumber = oneckpt["promptnumber"]
            model.promptembedding = oneckpt["promptembedding"]
            print("loaded model prompt weights")
        dooneeval(model, test_dataloader, test_result_dict, 0, output_dir, args, save_model=False)

    torch.cuda.empty_cache()
    del model, tokenizer
    gc.collect()

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()

    return result_dict


def dooneeval(modeltoeval, valid_dataloader, result_dict, i, path, args, save_model=True):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=args.stemmer)
    if isinstance(modeltoeval, torch.nn.parallel.DistributedDataParallel):
        model = modeltoeval.module
    else:
        model = modeltoeval
    model.eval()
    allentnumintar, allentnuminpre, hasentnum = 0, 0, 0
    alltar, allpred = [], []
    with torch.no_grad():
        logger.info(len(valid_dataloader))
        for step, batch in tqdm(enumerate(valid_dataloader)):
            #logger.info(step)
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
            sen, target, preds = model._generative_step(inputs)
            sennum = len(sen)
            for ii in range(sennum):
                thissen, thistar, thispred = sen[ii], target[ii], preds[ii]
                if thistar == 'end':
                    continue
                allentintar = thistar.lower().split(',')
                alleninpred = thispred.lower().split(',')

                allentnumintar += len(allentintar)
                allentnuminpre += len(alleninpred)
                for j in range(len(allentintar)):
                    if allentintar[j] in alleninpred:
                        hasentnum += 1
                alltar.append(thistar)
                allpred.append(thispred)
    if allentnuminpre != 0 and allentnumintar != 0:
        p = float(hasentnum) / float(allentnuminpre)
        r = float(hasentnum) / float(allentnumintar)
        if p + r != 0.0:
            f1score = 2 * p * r / (p + r)
        else:
            f1score = 0.0
    else:
        f1score = 0.0
    r1s, r2s, rls = [], [], []
    for j in range(len(alltar)):
        tar = alltar[j]
        pred = allpred[j]
        rouge_score = scorer.score(tar, pred)
        r1s.append(rouge_score["rouge1"].fmeasure)
        r2s.append(rouge_score["rouge2"].fmeasure)
        rls.append(rouge_score["rougeLsum"].fmeasure)
    r1 = np.mean(r1s)
    r2 = np.mean(r2s)
    rl = np.mean(rls)
    logger.info('----Validation Results Summary----')
    logger.info(f1score)
    logger.info(r1)
    logger.info(r2)
    logger.info(rl)

    result_dict['val_F1'].append(f1score)
    result_dict['val_r1'].append(r1)
    result_dict['val_r2'].append(r2)
    result_dict['val_rl'].append(rl)
    if result_dict['val_F1'][-1] > result_dict['best_val_F1']:
        logger.info("{} epoch, best epoch was updated! valid_F1: {: >4.5f}".format(i, result_dict['val_F1'][-1]))
        result_dict["best_val_F1"] = result_dict['val_F1'][-1]
        meanR = (r1 + r2 + rl) / 3
        result_dict["best_val_meanR"] = meanR

        if save_model:
            if not os.path.exists(path):
                os.mkdir(path)
            model_to_save = model.module if hasattr(model, 'module') else model
            if args.tune_weights:
                d = model_to_save.state_dict()
                d["promptnumber"] = model_to_save.promptnumber
                path = os.path.join(path, "bestckpt_full_weights")
                if args.use_pretrain_ckpt:
                    path += "_from_pretrained"
                torch.save(d, path)
            else:
                ckpt = {
                    "promptnumber": model_to_save.promptnumber,
                    "promptembedding": model_to_save.promptembedding
                }
                path = os.path.join(path, "bestckpt_prompt")
                if args.use_pretrain_ckpt:
                    path += "_from_pretrained"
                torch.save(ckpt, path)
            print("saved new entity model ckpt!")



