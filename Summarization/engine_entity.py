import spacy
import gc
import torch.nn as nn 
from datasets import load_metric
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
from transformers.optimization import Adafactor
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, PegasusConfig, PegasusTokenizerFast
gc.enable()

from utils import *
from dataset.dataset_pretrain import DatasetPretrain
from dataset.dataset_entity import *
from models.model_summary_soft import ModelSummarySoft
from models.model_entity import ModelEntity
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
    model = ModelEntity(basemodel, tokenizer, args)
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
            allckpt = torch.load("../support_files/conll_bestckpt")
            model.promptnumber = allckpt["promptnumber"]
            model.promptembedding = allckpt["promptembedding"]
        else:
            print("Initializing from scratch!")
            promptnumber = args.prompt_number
            taskname = "name entity recognition"
            promptembedding = getpromptembedding(model, tokenizer, promptnumber, taskname)
            print("prompt", promptembedding.shape)
            model.set_prompt_embedding(promptnumber, promptembedding)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("The model has {} trainable parameters".format(n_params))
    model.to(args.device)

    print(trainfile)
    print(max_seq_length)
    print(tokenizer)
    train_dataset = DatasetPretrain(trainfile, max_seq_length, tokenizer)
    valid_dataset = DatasetPretrain(validfile, max_seq_length, tokenizer)
    test_dataset = DatasetPretrain(testfile, max_seq_length, tokenizer)

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

    taggerfolder = "entity_ckpt/"
    os.makedirs(taggerfolder, exist_ok=True)
    output_dir = taggerfolder + args.dataset + "/" + str(args.few_shot) + "/" + seedname
    os.makedirs(output_dir, exist_ok=True)
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
        if not(args.zero_shot):
            if not (args.no_finetuned_eprompt):
                onepath = os.path.join(output_dir, f"bestckpt_prompt_{args.prompt_number}")
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
            ckpt = {
                "promptnumber": model_to_save.promptnumber,
                "promptembedding": model_to_save.promptembedding
            }
            path = os.path.join(path, f"bestckpt_prompt_{args.prompt_number}")
            if args.use_pretrain_ckpt:
                path += "_from_pretrained"
            torch.save(ckpt, path)
            print("saved new entity model ckpt!")


def infer_entity_model(alldata, enttokenizer, entmodel, args):
    allresofvalid = {}
    alltexts, allpreds, alllabels = [], [], []
    spacy_nlp = spacy.load("en_core_web_sm")
    count = 0
    with torch.no_grad():
        for step in tqdm(range(len(alldata))):
            onedata = alldata[step]
            inputdata = onedata[0]
            tempdata = re.sub(' +', ' ', inputdata).strip()
            alltexts.append(tempdata)
            inputres = enttokenizer.batch_encode_plus([tempdata], padding=True, max_length=args.max_length, truncation=True, return_tensors="pt")
            input_ids = inputres["input_ids"].to(args.device)
            attention_mask = inputres["attention_mask"].to(args.device)
            input = {"input_ids": input_ids, "attention_mask": attention_mask}
            _, _, tagpreds = entmodel._generative_step(input)
            allentitylist = tagpreds[0].split(',')
            if allentitylist == []:
                allentitylist = ["none"]
            input_guidance = args.separator.join(list(dict.fromkeys(allentitylist)))
            allresofvalid[tempdata] = input_guidance
            allpreds.append(tagpreds[0])
            target = onedata[1]
            ents = spacy_nlp(target).ents
            ents = [ent.text for ent in ents]
            target_ents = ','.join(ents)
            alllabels.append(target_ents)
            count += input_ids.shape[0]
            if count >= args.max_test_size:
                print("Hit the max test size...")
                break
    # ROUGE
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=args.stemmer)
    mean_rs, r1s, r2s, rls = [], [], [], []
    for i in range(len(allpreds)):
        rouge_score = scorer.score(alllabels[i], allpreds[i])
        r1 = rouge_score["rouge1"].fmeasure
        r2 = rouge_score["rouge2"].fmeasure
        rl = rouge_score["rougeLsum"].fmeasure
        mean_r = (r1 + r2 + rl) / 3
        mean_rs.append(mean_r)
        r1s.append(r1)
        r2s.append(r2)
        rls.append(rl)
    print("Entity inference mean R: {:.4f}, R-1: {:.4f}, R-2: {:.4f}, R-L: {:.4f}".format(
        100 * np.mean(mean_rs), 100 * np.mean(r1s), 100 * np.mean(r2s), 100 * np.mean(rls)
    ))
    # Precision, Recall, F-1
    ps, rs, f1s = [], [], []
    for i in range(len(allpreds)):
        preds_i = allpreds[i].lower().split(",")
        labels_i = alllabels[i].lower().split(",")
        p = 0.0
        r = 0.0
        for j in range(len(preds_i)):
            if preds_i[j] in labels_i:
                p += 1
        p /= len(preds_i)
        for j in range(len(labels_i)):
            if labels_i[j] in preds_i:
                r += 1
        r /= len(labels_i)
        if p + r != 0.0:
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0.0
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
    print("Entity inference Precision: {:.4f}, Recall: {:.4f}, F-1: {:.4f}".format(
        100 * np.mean(ps), 100 * np.mean(rs), 100 * np.mean(f1s)
    ))
    # abstractiveness
    new_pred_ents, new_target_ents = [], []
    for i in range(len(allpreds)):
        text = alltexts[i]
        text_words = word_tokenize(text)
        pred_entities = allpreds[i]
        pred_entities = pred_entities.split(",")
        new_pred = 100 * len([x for x in pred_entities if not(x in text_words)]) / max(1, len(pred_entities))
        new_pred_ents.append(new_pred)
        target_entities = alllabels[i]
        target_entities = target_entities.split(",")
        new_target = 100 * len([x for x in target_entities if not(x in text_words)]) / max(1, len(target_entities))
        new_target_ents.append(new_target)
    print("Abstractive predicted entities: {:.4f}% | target entities: {:.4f}%".format(
        np.mean(new_pred_ents), np.mean(new_target_ents)
    ))
    # repetition
    rep_preds, rep_targets = [], []
    for i in range(len(allpreds)):
        pred_entities = allpreds[i]
        pred_entities = pred_entities.split(",")
        rep = 0
        for j in range(1, len(pred_entities)):
            if pred_entities[j] in pred_entities[:j]:
                rep += 1
        rep /= max(1, len(pred_entities)-1)
        rep *= 100
        rep_preds.append(rep)
        target_entities = alllabels[i]
        target_entities = target_entities.split(",")
        rep = 0
        for j in range(1, len(target_entities)):
            if target_entities[j] in target_entities[:j]:
                rep += 1
        rep /= max(1, len(target_entities)-1)
        rep *= 100
        rep_targets.append(rep)
    print("Repeated predicted entities: {:.4f}% | target entities: {:.4f}%".format(
        np.mean(rep_preds), np.mean(rep_targets)
    ))

    return allresofvalid, allpreds, alllabels, mean_rs

