import numpy as np 
import spacy

from tqdm import tqdm 
from torch.utils import data
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor
from torch.cuda.amp import autocast as autocast
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from datasets import load_metric

from utils import *
from dataset import *
from model import *
from model_finetune import T5Finetune
from model_mixture import T5MixPrompt
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from prompting import *
from rouge_score import rouge_scorer



def load_model(args):
    # base model & tokenizer (use T5)
    t5model = T5ForConditionalGeneration.from_pretrained(args.model_name,cache_dir=args.cache_dir)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name,cache_dir=args.cache_dir)

    # model
    if args.model == "T5Prompt":
        model = T5Prompt(args,t5model,tokenizer)
        if args.ckpt_path and args.load_ckpt:
            load_prompt(args, model)
        else:
            prompt_length = args.prompt_length
            prompt_embedding = get_prompt_embedding(model, tokenizer, prompt_length)
            model.set_prompt_embedding(prompt_length, prompt_embedding)
        model.to(args.device)
    elif args.model == 'T5MixPrompt':
        model = T5MixPrompt(args, t5model, tokenizer)
        if args.ckpt_path and args.load_ckpt:
            load_prompt(args, model)
            model.to(args.device)
        else:
            label_name_embs = get_mix_prompt_embedding(model, tokenizer, args.prompt_length, args.prompt_length_discrete)
            model.to(args.device)
            model.set_prompt_embedding(label_name_embs)
    elif args.model == 'T5Finetune':
        model = T5Finetune(args, t5model, tokenizer)
        if args.ckpt_path and args.load_ckpt:
            load_prompt(args, model)
        model.to(args.device)
    else:
        raise Exception("No such model! Please make sure that `model` takes the value in {T5}")
    return model, tokenizer


def train(args, model, train_dataset, valid_dataset, test_dataset, logger):
    # total step
    step_tot = int(np.round((0.5 + (train_dataset.num_entries / (float(args.gradient_accumulation_steps) * args.batch_size_per_gpu * args.n_gpu))))) * args.max_epoch

    train_sampler = data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(train_dataset)

    valid_sampler = SequentialSampler(valid_dataset)

    train_dataloader = get_dataloader(
        args.num_workers, train_dataset, args.batch_size_per_gpu, args.max_length, args.max_guidance_length, args.max_summary_length, 
        train_dataset.tokenizer.pad_token_id, train_sampler
    )
    valid_dataloader = get_dataloader(
        args.num_workers, valid_dataset, args.valid_size_per_gpu, args.max_length, args.max_guidance_length, args.max_summary_length, 
        valid_dataset.tokenizer.pad_token_id, valid_sampler
    )

    base_optimizer_arguments = {"lr": args.lr, "clip_threshold": args.max_grad_norm, "decay_rate": -0.8,
                                "weight_decay": args.weight_decay,
                                "scale_parameter": False, "relative_step": False}
    if args.model == 'T5Finetune':
        optimizer = AdamW
        base_optimizer_arguments = {"lr": args.lr, "weight_decay": args.weight_decay}
    else:
        optimizer = Adafactor
    if args.n_gpu > 1: # distributed training
        optimizer = OSS(params=filter(lambda p: p.requires_grad, model.parameters()), optim=optimizer,
                        **base_optimizer_arguments)
        # distributed training
        model = ShardedDDP(model, optimizer)
    else:
        optimizer = optimizer(params=filter(lambda p: p.requires_grad, model.parameters()), **base_optimizer_arguments)
    #import pdb;pdb.set_trace() #len(optimizer.param_groups[0]['params']) #len(optimizer.param_groups)
    model.train()
    #scaler = ShardedGradScaler()
    scheduler = None
    scaler = None

    startepoch = 0
    Best_F1 = 0.0

    logger.info("Begin train...")
    logger.info("We will train model in %d steps" % step_tot)

    result_dict = {
        'epoch': [],
        'val_mean_rouge': [],
        'best_val_mean_rouge': 0.0,
        'val_rouge1': [],
        'val_rouge2': [],
        'val_rougeL': []
    }
    global_step = 0
    model.eval()
    model.train()
    alllosses=[]
    logger.info("Epoch 0 validation")
    dooneeval(args, model, valid_dataloader, scaler, result_dict, logger, 0)
    model.train()
    for i in range(startepoch, startepoch + args.max_epoch):
        logger.info(">>>>> New epoch, epoch {} / {}".format(i + 1, args.max_epoch))
        adjusted_evalstep = args.eval_step

        model.train()
        result_dict['epoch'] = i
        allloss = []
        for step, batch in enumerate(train_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device),
                      "input_ents": batch[4].to(args.device), "ents_mask": batch[5].to(args.device)}

            if scaler is not None:
                with autocast():
                    loss = model(inputs)
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
            else:
                loss = model(inputs)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
            finalloss = loss 
            if scaler is not None:
                scaler.scale(finalloss).backward()
            else:
                finalloss.backward()
            allloss.append(loss.item())
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if scaler is not None:
                    #scaler.unscale_(optimizer)
                    #optimizer.clip_grad_norm(args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    #nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                if scheduler != None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [0, -1] and global_step % args.log_step == 0:
                    #logger.info("step: %d, shcedule: %.3f, loss: %.6f" % (global_step, global_step/step_tot, np.average(allloss)))
                    logger.info("step: %d, schedule: %.3f, loss: %.6f, epoch: %d" % (
                    global_step, global_step / step_tot, np.average(allloss) * args.gradient_accumulation_steps, i))

                if args.local_rank in [0, -1] and global_step % adjusted_evalstep == 0:
                    dooneeval(args, model, valid_dataloader, scaler, result_dict, logger, i)
                    model.train()

        mean_loss = np.mean(allloss)
        logger.info("Mean training loss: {:.4f}".format(mean_loss))
        if args.local_rank in [0, -1]:
            dooneeval(args, model, valid_dataloader, scaler, result_dict, logger, i)
            model.train()

    logger.info('finish training')
    
    return result_dict


def dooneeval(args, model, valid_dataloader, scaler, result_dict, logger, i):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    else:
        model = model
    model.eval()
    allytrue = []
    allypred = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(valid_dataloader)):            
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device), "input_ents": batch[4].to(args.device), "ents_mask": batch[5].to(args.device)}
            if scaler is not None:
                with autocast():
                    sen, target, preds, _ = model._generative_step(inputs)
                    tarres, predres = target, preds
                    allytrue.extend(tarres)
                    allypred.extend(predres)
            else:
                sen, target, preds, _ = model._generative_step(inputs)
                tarres, predres = target, preds
                allytrue.extend(tarres)
                allypred.extend(predres)
    # 1st method
    #rouge = load_metric('rouge')
    #rouge_score = rouge.compute(references=[x.lower() for x in allytrue], predictions=[x.lower() for x in allypred])
    #for k in rouge_score.keys():
    #    rouge_score[k] = 100 * rouge_score[k].mid.fmeasure
    # 2nd method
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer = False)
    r1s, r2s, rls = [], [], []
    for i in range(len(allytrue)):
        label = allytrue[i]
        summary = allypred[i]
        label = "\n".join(sent_tokenize(label))
        summary = "\n".join(sent_tokenize(summary))
        rouge_score = scorer.score(label, summary)
        r1s.append(rouge_score["rouge1"].fmeasure)
        r2s.append(rouge_score["rouge2"].fmeasure)
        rls.append(rouge_score["rougeLsum"].fmeasure)
    rouge_score = {
        "rouge1": 100 * np.mean(r1s),
        "rouge2": 100 * np.mean(r2s),
        "rougeLsum": 100 * np.mean(rls)
    }
    logger.info('----Validation Results Summary----')
    logger.info(len(allypred))
    logger.info(rouge_score)
    p, r, f1 = entity_eval(allytrue, allypred)

    # result_dict['val_rouge1'].append(rouge_score["rouge1"].mid.fmeasure)
    # change accordingly
    mean_rouge = (rouge_score["rouge1"] + rouge_score["rouge2"] + rouge_score["rougeLsum"]) / 3
    result_dict['val_mean_rouge'].append(mean_rouge)
    if result_dict['val_mean_rouge'][-1] > result_dict['best_val_mean_rouge']:
        logger.info("{} epoch, best epoch was updated! val_mean_rouge: {: >4.5f}".format(i, result_dict['val_mean_rouge'][-1]))
        result_dict["best_val_mean_rouge"] = result_dict['val_mean_rouge'][-1]
        # also append other rouge scores
        result_dict['val_rouge1'] = rouge_score["rouge1"]
        result_dict['val_rouge2'] = rouge_score["rouge2"]
        result_dict['val_rougeL'] = rouge_score["rougeLsum"]
        
        result_dict['precision'] = p
        result_dict['recall'] = r
        result_dict['f1'] = f1

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
        torch.save(ckpt, os.path.join(args.save_path + "/" + args.save_dir, "ckptofT5_best"))
        logger.info("ckpt saved")


def test(args, test_dataset, tokenizer, logger):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = get_dataloader(
        args.num_workers, test_dataset, args.test_size_per_gpu, args.max_length, args.max_guidance_length, args.max_summary_length, 
        test_dataset.tokenizer.pad_token_id, test_sampler
    )

    t5model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    allckpt = torch.load(args.save_path + "/" + args.save_dir + "/ckptofT5_best")
    if args.model == 'T5Prompt':
        model = T5Prompt(args, t5model, tokenizer)
        model.prompt_length = allckpt["prompt_length"]
        model.prompt_embedding = allckpt["prompt_embedding"]
    elif args.model == 'T5MixPrompt':
        model = T5MixPrompt(args, t5model, tokenizer)
        model.prompt_dict = allckpt['prompt_dict']
        model.prompt_fix_dict = allckpt['prompt_fix_dict']
    elif args.model == 'T5Finetune':
        model = T5Finetune(args, t5model, tokenizer)
        #model_state_dict = {}
        #for k,v in allckpt['t5-base'].items():
        #    model_state_dict['model.'+k] = v
        #model.load_state_dict(model_state_dict)
        model.model.load_state_dict(allckpt["t5-base"])
    logger.info("load finished!")

    model.to(args.device)
    model.eval()
    allytrue = []
    allypred = []
    #scaler = ShardedGradScaler()
    scaler = None

    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_dataloader)):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device), "input_ents": batch[4].to(args.device), "ents_mask": batch[5].to(args.device)}
            if scaler is not None:
                with autocast():
                    source, target, preds, ents = model._generative_step(inputs)
                    tarres, predres = target, preds
                    allytrue.extend(tarres)
                    allypred.extend(predres)
            else:
                # print(f"eval step: {step}")
                source, target, preds, ents = model._generative_step(inputs)
                tarres, predres = target, preds
                allytrue.extend(tarres)
                allypred.extend(predres)

            if args.display_preds:
                if step < 5:
                    display_preds(source, target, preds, ents)

    rouge = load_metric('rouge')
    rouge_score = rouge.compute(references=allytrue, predictions=allypred)
    logger.info('-----Test Results Summary-----')
    logger.info(len(allypred))
    logger.info(rouge_score)
    p, r, f1 = entity_eval(allytrue, allypred)
    result_dict = {}
    result_dict['test_rouge1'] = rouge_score["rouge1"].mid.fmeasure
    result_dict['test_rouge2'] = rouge_score["rouge2"].mid.fmeasure
    result_dict['test_rougeL'] = rouge_score["rougeL"].mid.fmeasure
    result_dict['precision'] = p
    result_dict['recall'] = r
    result_dict['f1'] = f1
    
    return result_dict


def display_preds(source, target, preds, ents):
    print("\nDisplaying a new batch of results:")
    for i in range(len(source)):
        print("Source:")
        print(source[i])
        print("Entities:")
        print(ents[i])
        print("Reference:")
        print(target[i])
        print("Predicted summary:")
        print(preds[i])


def entity_eval(ytrue, ypred):
    spacy_nlp = spacy.load("en_core_web_sm")
    all_p = []
    all_r = []
    all_f1 = []
    for i in tqdm(range(len(ytrue))):
        ents_true = spacy_nlp(ytrue[i]).ents
        ents_true = [ent.text for ent in ents_true]
        ents_pred = spacy_nlp(ypred[i]).ents
        ents_pred = [ent.text for ent in ents_pred]
        p = 0
        r = 0
        f1 = 0
        if len(ents_pred) > 0:
            p = 100 * len([x for x in ents_pred if x in ents_true]) / len(ents_pred)
        else:
            if len(ents_true) == 0:
                p = 100
        if len(ents_true) > 0:
            r = 100 * len([x for x in ents_true if x in ents_pred]) / len(ents_true)
        else:
            if len(ents_pred) == 0:
                r = 100
        if (p + r) > 0:
            f1 = (2 * p * r) / (p + r)
        all_p.append(p)
        all_r.append(r)
        all_f1.append(f1)
    p = np.mean(all_p)
    r = np.mean(all_r)
    f1 = np.mean(all_f1)
    print("\nEntity-level eval, mean precision: {:.4f}, recall: {:.4f}, F-1: {:.4f}".format(p, r, f1))
    
    return p, r, f1
