from torch.utils import data
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor
from torch.cuda.amp import autocast as autocast
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from datasets import load_metric

from model import *
from model_finetune import T5Finetune
from model_mixture import T5MixPrompt
from dataset import *
from utils import *




def dooneeval(modeltoeval, valid_dataloader, args, result_dict, optimizer, scaler, i, logger):
    if isinstance(modeltoeval, torch.nn.parallel.DistributedDataParallel):
        model = modeltoeval.module
    else:
        model = modeltoeval
    model.eval()
    allytrue = []
    allypred = []
    with torch.no_grad():
        logger.info(len(valid_dataloader))
        for step, batch in enumerate(valid_dataloader):
            #logger.info(step)
            
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device), "input_ents": batch[4].to(args.device), "ents_mask": batch[5].to(args.device)}
            if scaler is not None:
                with autocast():
                    sen, target, preds = model._generative_step(inputs)
                    tarres, predres = target, preds
                    allytrue.extend(tarres)
                    allypred.extend(predres)
            else:
                # print(f"eval step: {step}")
                sen, target, preds = model._generative_step(inputs)
                tarres, predres = target, preds
                allytrue.extend(tarres)
                allypred.extend(predres)
    rouge = load_metric('rouge')
    rouge_score = rouge.compute(references=allytrue, predictions=allypred)
    logger.info('----Validation Results Summary----')
    logger.info(len(allypred))
    logger.info(rouge_score)

    result_dict['val_rouge1'].append(rouge_score["rouge1"].mid.fmeasure)
    if result_dict['val_rouge1'][-1] > result_dict['best_val_rouge1']:
        logger.info("{} epoch, best epoch was updated! val_rouge1: {: >4.5f}".format(i,result_dict['val_rouge1'][-1]))
        result_dict["best_val_rouge1"] = result_dict['val_rouge1'][-1]
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
        torch.save(ckpt, os.path.join(args.save_path + "/" + args.save_dir, "ckptofT5_best"))
        print("ckpt saved")

def test(args, test_dataset, logger, tokenizer):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = get_dataloader(args.num_workers, test_dataset, args.test_size_per_gpu, args.max_length, args.max_guidance_len,
                                      args.max_target_length, test_dataset.tokenizer.pad_token_id,test_sampler)

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
        model_state_dict = {}
        for k,v in allckpt['t5-base'].items():
            model_state_dict['model.'+k] = v
        model.load_state_dict(model_state_dict)
    logger.info("load finished!")

    model.to(args.device)
    model.eval()
    allytrue = []
    allypred = []
    #scaler = ShardedGradScaler()
    scaler = None

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device), "input_ents": batch[4].to(args.device), "ents_mask": batch[5].to(args.device)}
            if scaler is not None:
                with autocast():
                    sen, target, preds = model._generative_step(inputs)
                    tarres, predres = target, preds
                    allytrue.extend(tarres)
                    allypred.extend(predres)
            else:
                # print(f"eval step: {step}")
                sen, target, preds = model._generative_step(inputs)
                tarres, predres = target, preds
                allytrue.extend(tarres)
                allypred.extend(predres)
    rouge = load_metric('rouge')
    rouge_score = rouge.compute(references=allytrue, predictions=allypred)
    logger.info('-----Test Results Summary-----')
    logger.info(len(allypred))
    logger.info(rouge_score)


def train(args, model, train_dataset, valid_dataset, test_dataset, logger):
    # total step
    step_tot = int(0.5 + train_dataset.num_entries / float(args.gradient_accumulation_steps) / args.batch_size_per_gpu / args.n_gpu) * args.max_epoch

    warmup_steps_total = step_tot * args.warmup_steps
    train_sampler = data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(train_dataset)

    valid_sampler = SequentialSampler(valid_dataset)

    train_dataloader = get_dataloader(args.num_workers, train_dataset, args.batch_size_per_gpu, args.max_length, args.max_guidance_len,
                                      args.max_target_length, train_dataset.tokenizer.pad_token_id,train_sampler)
    valid_dataloader = get_dataloader(args.num_workers, valid_dataset, args.valid_size_per_gpu, args.max_length, args.max_guidance_len,
                                      args.max_target_length, valid_dataset.tokenizer.pad_token_id,valid_sampler)


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
        'val_rouge1': [],
        'best_val_rouge1': Best_F1
    }
    global_step = 0
    model.eval()
    model.train()
    for i in range(startepoch, startepoch + args.max_epoch):
        print("\n>>>>>>>>>>>>>>New epoch<<<<<<<<<<<<<<\n")
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
                    dooneeval(model,valid_dataloader,args,result_dict,optimizer,scaler,i)
                    model.train()
                    print('back to train')


                if args.local_rank in [0, -1] and global_step % args.save_step == 0:
                    save_model(model, args, global_step)
                    model.train()

        print("\nEnd of epoch evaluation...")
        dooneeval(model, valid_dataloader, args, result_dict, optimizer, scaler, i, logger)
        save_model(model, args, global_step)
        model.train()
        print('back to train')

        if args.train_sample:
            logger.info("sampling...")
            logger.info("sampled")
    print('finish training')
    if args.local_rank in [0, -1]:
        save_model(model, args, global_step)



