# This script performs qualitative tests to test a model for controllablity
import argparse
import logging
from utils import Nop
import torch
from dataset_finetune_entity import *
from dataset_finetune_summary import *
from engine_pretrain import *
from engine_finetune_entity import *
from engine_finetune_summary import *
from models_summarization.model_mixture import *
from nltk.tokenize import sent_tokenize

def set_args():
    parser = argparse.ArgumentParser(description="latentRE")
    data_root = "/data/ruochen/"
    root = "/data/mathieu/"
    parser.add_argument("--data_dir", dest="data_dir", type=str,
                        default= data_root + "DATASETS/PromptSumm/")
    parser.add_argument("--dataset_name", dest="dataset_name", type=str,
                            default="ccdv/cnn_dailymail")
    parser.add_argument("--model", dest="model", type=str,
                        default="T5MixPrompt", choices = ["T5Finetune", "T5SoftPrompt", "T5MixPrompt",
                            "BartFinetune", 'BartSoftPrompt', 'BartMixPrompt'])
    parser.add_argument("--model_name", dest="model_name", type=str,
                        default="google/t5-v1_1-large", help="{t5-base, google/t5-v1_1-base, facebook/bart-base, facebook/bart-large}")
    parser.add_argument("--use_lm_adapted", dest="use_lm_adapted", type=int,
                        default=1, help="whether to use lm_adapted model") #if we use bart, then automatically don't use lm_adapted
    parser.add_argument("--lm_adapted_path", dest="lm_adapted_path", type=str,
                        default=root + "lm_adapted_t5model/torch_ckpt/large/pytorch_model.bin",
                        help="The path of lm_adapted model")
    parser.add_argument("--cache_path", dest="cache_path", type=str,
                        default=root + "hf_models/t5-v1-large/",
                        help="The path of huggingface cache") # /data/ruochen/hf_models/bart-base for bart
    parser.add_argument("--dataset_cache_dir", dest="dataset_cache_dir", type=str,
                        default="../../hf_datasets/", help="dataset cache folder")
    parser.add_argument("--guidance_type", dest="guidance_type", type=str,
                        default="ents")
    parser.add_argument("--separator", dest="separator", type=str,
                        default=",", choices=[",", " "])
    parser.add_argument("--guidance_mode", dest="guidance_mode", type=str,
                        default="input", choices=["input", "input_most_frequent", "input_salient_sentences", "input_and_target", "target", "target_unique", 'target_unique_filtered'])
    parser.add_argument("--log_dir", dest="log_dir", type=str,
                            default='./log', help="The path to log dir")
    parser.add_argument("--save_model_path", dest="save_model_path", type=str,
                            default='/data/ruochen/DATASETS/PromptSumm/xsum/10/seed_0/best_ckpt', help="The path to log dir")
    parser.add_argument("--log_name", dest="log_name", type=str,
                        default='controlling', help="The file name of log file")
    parser.add_argument("--num_workers_summary", dest="num_workers_summary", type=int,
                        default=0, help="dataloader num_workers")
    parser.add_argument("--valid_size_per_gpu_summary", dest="valid_size_per_gpu_summary", type=int,
                        default=4, help="valid size per gpu")
    parser.add_argument("--max_length", dest="max_length", type=int,
                        default=512, help="max sentence length")
    parser.add_argument("--max_guidance_length", dest="max_guidance_length", type=int,
                        default=100)
    parser.add_argument("--local_rank", dest="local_rank", type=int,
                        default=-1, help="local rank")
    parser.add_argument("--few_shot", dest="few_shot", type=int,
                        default=10, help="number of data points for training AND validation")
    parser.add_argument("--use_t5_tagger",  action='store_false',
                        default=True, help="whether use a t5 tagger")
    parser.add_argument("--infer_val_entities", action="store_false",
                        default=True, help="whether to run inference with the T5 entity chain prediction on val set")
    parser.add_argument("--pretrain", action='store_true',
                        default=False, help="whether pretrain a T5 tagger")
    parser.add_argument("--num_beams", dest="num_beams", type=int,
                        default=4, help="number of beams in beam search")
    parser.add_argument("--repetition_penalty", dest="repetition_penalty", type=float,
                        default=2.5, help="repetition penalty")
    parser.add_argument("--length_penalty", dest="length_penalty", type=float,
                        default=1.0, help="length penalty")
    parser.add_argument("--stemmer", dest="stemmer", type=bool, 
                        default=True)
    parser.add_argument("--prompt_number", dest="prompt_number", type=int,
                        default=300, help="The number of prompt")
    parser.add_argument("--use_pretrain_ckpt", action='store_false',
                        default=True, help="whether to load the pre-training ckpt before fine-tuning")
    parser.add_argument("--pretrain_ckpt", type=str,
                        default="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_510k/bestckpt_full_model", help="path to pretrained model")
    parser.add_argument("--pretrain_prompt_ckpt", type=str,
                        default="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_510k/bestckpt_prompt", help="path to pretrained model prompt")
    
    dataset_names = ["ccdv/cnn_dailymail", "xsum", "reddit_tifu", "wikihow", "billsum", "samsum","c4"]
    
    args = parser.parse_args()
    ## SET HERE FOR PRETRAIN
    # args.pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_330k/bestckpt_full_model"
    # args.pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_330k/bestckpt_prompt"
    max_summary_lengths = [128, 64, 64, 128, 256, 64]
    highlights = [True, False, False, False, False, False, False]
    
    idx = dataset_names.index(args.dataset_name)
    if args.dataset_name == 'cnn_dailymail' or args.dataset_name == "ccdv/cnn_dailymail":
        idx = 0
        args.dataset = 'cnndm'
    else:
        args.dataset = args.dataset_name
    args.highlights = highlights[idx]
    args.max_summary_length = max_summary_lengths[idx]
    return args

def set_logger(args):
    global logger
    logger = logging.getLogger()
    logging.basicConfig(filename=args.log_name,
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
    fh = logging.FileHandler(args.log_name)
    logger.addHandler(fh)
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    #     datefmt = '%m/%d/%Y %H:%M:%S',
    #     handlers=[
    #         logging.FileHandler(f"{args.log_dir}/{args.log_name}.log"),
    #         logging.StreamHandler()
    #     ]
    # )

def eval(model, valid_dataset, scaler, logger, args, tokenizer, seed = 0):
    model.eval()
    model = model.to(args.device)
    all_ents = []
    if args.use_t5_tagger and args.model == "T5MixPrompt" and args.guidance_mode != "target":
        if args.infer_val_entities:
            ########## predict the validation entity chains with the 1st prompt tuning stage model
            entbasemodel = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir = args.cache_path)
            enttokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir = args.cache_path)
            entmodel = T5forFinetuneEntity(entbasemodel, enttokenizer, args)
            logger.info("Loading the pre-trained NER model!")

            # model weights
            ckpt = torch.load(args.pretrain_ckpt)
            dic = {}
            for x in ckpt.keys():
                if not (x in ["module.promptnumber", "module.promptembedding", "module.promptnumberforsum", "module.promptembeddingforsum"]):
                    dic[x[7:]] = ckpt[x]
            entmodel.load_state_dict(dic)

            # just prompt
            #onepath = f'{args.few_shot_save_dir}seed_{seed}/data_for_bert_{seed}/tagger/bestckpt_prompt' ####bestckpt_prompt?
            onepath = f'tagger_ckpt/{args.dataset}/{args.few_shot}/seed_{seed}/bestckpt_prompt'
            print(onepath)
            oneckpt = torch.load(onepath)
            entmodel.promptnumber = oneckpt["promptnumber"]
            entmodel.promptembedding = oneckpt["promptembedding"]
        
            n_params = sum(p.numel() for p in entmodel.parameters() if p.requires_grad)
            logger.info("The ent model has {} trainable parameters".format(n_params))
            entmodel.to(args.device)
            logger.info("move to device!")
            model.eval()

            alldata = valid_dataset.data
            #logger.info("valid size: ", len(alldata))
            print("valid size: ", len(alldata))
            allresofvalid = {}
            with torch.no_grad():
                for step in range(len(alldata)):
                    onedata = alldata[step]
                    inputdata = onedata[0]
                    tempdata = re.sub(' +', ' ', inputdata)
                    inputres = enttokenizer.batch_encode_plus([tempdata], padding=True, max_length=args.max_length, truncation=True, return_tensors="pt")
                    input_ids = inputres["input_ids"].to(args.device)
                    attention_mask = inputres["attention_mask"].to(args.device)
                    input = {"input_ids": input_ids, "attention_mask": attention_mask}
                    tagpreds = entmodel._generative_step_for_tagger(input)
                    allentitylist = tagpreds[0].split(',')
                    if allentitylist == []:
                        allentitylist = ["none"]
                    # if counterfactual_removal, remove them
                    input_guidance_list = list(dict.fromkeys(allentitylist))
                    if args.counterfactual_removal != False:
                        for c_r in range(int(args.counterfactual_removal)):
                            if len(input_guidance_list) > 2:
                                input_guidance_list.pop(random.randrange(len(input_guidance_list)))
                        input_guidance = args.separator.join(input_guidance_list)
                        allresofvalid[tempdata] = input_guidance
                    else:
                        input_guidance = args.separator.join(list(dict.fromkeys(allentitylist)))
                        allresofvalid[tempdata] = input_guidance
                    all_ents.append(input_guidance)
            logger.info(len(allresofvalid))
            respath = f'tagger_ckpt/{args.dataset}/{args.few_shot}/seed_{seed}/T5valident.pkl'
            with open(respath, "wb") as f:
                pickle.dump(allresofvalid, f)
                logger.info("saved the T5 valid entities")
            torch.cuda.empty_cache()
            del entmodel, enttokenizer
            gc.collect()
            valid_dataset.set_allent_for_valid()
    else:
        all_ents = []
        if args.counterfactual_removal != False:
            new_allent = {}
        for key, value in valid_dataset.allent.items():
            input_guidance_list = value.split(',')
            if args.counterfactual_removal != False:
                for c_r in range(int(args.counterfactual_removal)):
                    if len(input_guidance_list) > 2:
                        input_guidance_list.pop(random.randrange(len(input_guidance_list)))
            input_guidance = args.separator.join(input_guidance_list)
            all_ents.append(input_guidance)
            if args.counterfactual_removal != False:
                new_allent[key] = input_guidance
        if args.counterfactual_removal != False:
            valid_dataset.allent = new_allent
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = get_dataloader(tokenizer, args.num_workers_summary, valid_dataset, args.valid_size_per_gpu_summary, args.max_length,
                                      args.max_guidance_length, valid_dataset.tokenizer.pad_token_id, valid_sampler, args)
    all_inputs = []
    allytrue = []
    allypred = []
    with torch.no_grad():
        for step, batch in enumerate(valid_dataloader):
            # logger.info(step)
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device),
                      "ents_ids": batch[4].to(args.device), "ents_mask": batch[5].to(args.device),
                      "predents_ids": batch[6].to(args.device), "predents_mask": batch[7].to(args.device)}
            allinp = []
            for k in range(inputs["ents_ids"].shape[0]):
                # print(inputs['input_ids'][k])
                allinp.append(tokenizer.decode(inputs['input_ids'][k]))
            # print('allinp: ', allinp)
            all_inputs += allinp
            if scaler is not None:
                with autocast():
                    sen, target, preds = model._generative_step(inputs)
                    tarres, predres = target, preds
                    allytrue.extend(tarres)
                    allypred.extend(predres)
            else:
                sen, target, preds = model._generative_step(inputs)
                tarres, predres = target, preds
                allytrue.extend(tarres)
                allypred.extend(predres)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer = args.stemmer)
    r1s, r2s, rls = [], [], []
    labels = []
    summaries = []
    for j in range(len(allytrue)):
        label = allytrue[j]
        summary = allypred[j]
        if args.highlights:
            label = "\n".join(sent_tokenize(label))
            summary = "\n".join(sent_tokenize(summary))
        rouge_score = scorer.score(label, summary)
        labels.append(label)
        summaries.append(summary)
        r1s.append(rouge_score["rouge1"].fmeasure)
        r2s.append(rouge_score["rouge2"].fmeasure)
        rls.append(rouge_score["rougeLsum"].fmeasure)
    rouge_score = {
        "rouge1": 100 * np.mean(r1s),
        "rouge2": 100 * np.mean(r2s),
        "rougeLsum": 100 * np.mean(rls)
    }
    # logger.info(len(allypred))
    # logger.info(rouge_score)
    p, r, f1 = entity_eval(allytrue, allypred)
    logger.info(f'----Validation Results Summary counterfactual_removal: {args.counterfactual_removal}----')
    logger.info(f'entity-level p: {p}, r: {r}, f1: {f1}')
    logger.info(f'ROUGE score r1: {rouge_score["rouge1"]}, r2: {rouge_score["rouge2"]}, r3: {rouge_score["rougeLsum"]}')
    mean_rouge = (rouge_score["rouge1"] + rouge_score["rouge2"] + rouge_score["rougeLsum"]) / 3
    logger.info(f'mean-rouge: {mean_rouge}')
    logger.info(f'----EVAL FINISHED----')
    # print('INPUTS: ', inputs)
    return all_inputs, all_ents, labels, summaries


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    args.device = device
    logger.info(f"device {args.device}")
    allgentasktokens = [f"summerization{args.dataset}"]
    thistaskname = "cnn daily mail" if args.dataset=='cnndm' else args.dataset
    thistaskfold = args.dataset
    args.taskfold = thistaskfold
    # First load the trained ckpt
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)
    basemodel = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
    model = ModelMixPrompt(args, basemodel, tokenizer, args.model)
    promptnumber = args.prompt_number
    promptembedding = getpromptembedding(model, tokenizer, promptnumber, thistaskname)
    model.set_prompt_embedding(promptnumber, promptembedding)
    # model weights
    if args.use_pretrain_ckpt and args.model != "T5Finetune":
        ckptsum = torch.load(args.pretrain_ckpt)
        dicsum = {}
        for x in ckptsum.keys():
            if not (x in ["module.promptnumberforsum", "module.promptembeddingforsum"]):
                dicsum[x[7:]] = ckptsum[x]
        model.load_state_dict(dicsum)
    seed = 0
    args.few_shot_save_dir = args.data_dir + args.dataset + "/{}/".format(args.few_shot)
    ckptsum = torch.load(args.save_model_path)
    model.promptnumber = ckptsum["promptnumber"]
    model.promptembedding = nn.parameter.Parameter(ckptsum["promptembedding"])
    few_shot_seeds = [0]
    for gg in range(len(allgentasktokens)):
        gentasktoken = allgentasktokens[gg]
        tokenizer.add_tokens(gentasktoken)
        logger.info('gen token = {} , gen token id = {}'.format(
            gentasktoken, tokenizer.convert_tokens_to_ids(gentasktoken)
        ))
    answertoken = "__ans__"
    special_tokens = {"ans_token": answertoken}
    tokenizer.add_tokens(list(special_tokens.values()))
    tokenizer.add_tokens(['[SEP]'])
    # For each sentence
    # predict as is
    args.counterfactual_removal = False
    valid_file_name = args.few_shot_save_dir + 'seed_{}/valid.txt'.format(0)
    valid_dataset = T5SummarizationDataset(valid_file_name, "valid", args.max_length, tokenizer, allgentasktokens, answertoken, args, 0)
    scaler = None
    inputs0, all_ents0, labels0, summaries0 = eval(model, valid_dataset, scaler, logger, args, tokenizer)
    # remove one entity
    args.counterfactual_removal = 1
    valid_file_name = args.few_shot_save_dir + 'seed_{}/valid.txt'.format(0)
    valid_dataset = T5SummarizationDataset(valid_file_name, "valid", args.max_length, tokenizer, allgentasktokens, answertoken, args, 0)
    scaler = None
    inputs1, all_ents1, labels1, summaries1 = eval(model, valid_dataset, scaler, logger, args, tokenizer)
    # remove two entities
    args.counterfactual_removal = 2
    valid_file_name = args.few_shot_save_dir + 'seed_{}/valid.txt'.format(0)
    valid_dataset = T5SummarizationDataset(valid_file_name, "valid", args.max_length, tokenizer, allgentasktokens, answertoken, args, 0)
    scaler = None
    inputs2, all_ents2, labels2, summaries2 = eval(model, valid_dataset, scaler, logger, args, tokenizer)
    for i in range(len(labels0)):
        logger.info('-----')
        logger.info(f'INPUT: {inputs0[i]}')
        logger.info(f'LABEL: {labels0[i]}')
        logger.info(f'ENTS: {all_ents0[i]}; SUMMARY: {summaries0[i]}')
        logger.info(f'ENTS: {all_ents1[i]}; SUMMARY: {summaries1[i]}')
        logger.info(f'ENTS: {all_ents2[i]}; SUMMARY: {summaries2[i]}')
    # insert one entity
    # insert two entities
if __name__ == "__main__":
    args = set_args()
    set_logger(args)
    logger.info(args)
    main(args)


