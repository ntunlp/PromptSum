# This script performs qualitative tests to test a model for controllablity: success rates
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
from pathlib import Path
from rouge_score import rouge_scorer
import random
import copy
import json
from dataset import *
from nltk.tokenize import sent_tokenize
from spacy.lang.en import English
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast
from collections import defaultdict
from fairseq.models.bart import BARTModel
from itertools import combinations

def set_args():
    parser = argparse.ArgumentParser(description="latentRE")
    #root = "/export/home/"
    #data_root = "/export/home/"
    root = "/data/mathieu/"
    data_root = "/data/mathieu/"
    parser.add_argument("--data_dir", dest="data_dir", type=str,
                        default= data_root + "DATASETS/PromptSumm/")
    parser.add_argument("--CTRLsum_ckpt_dir", dest="CTRLsum_ckpt_dir", type=str,
                        default='/export/home/ctrl-sum/cnndm_ctrlsum_100')
    parser.add_argument("--mode", dest="mode", type=str,
                        default="single_entity_test", choices = ["single_entity_test", "oracle_add_test", "oracle_drop_test", "oracle", "k_entity_test", "interactive"])
    parser.add_argument("--k_entity", dest="k_entity", type=int,
                        default=2)                    
    parser.add_argument("--single_word", action='store_true',
                        default=False, help="whether to filter out multiple token entities")
    parser.add_argument("--cuda", dest="cuda", type=str,
                        default="2", help="gpu id") 
    parser.add_argument("--tune_weights", dest="tune_weights", action='store_true',
                        default=False)
    parser.add_argument("--ckpt_name", dest="ckpt_name", type=str,
                        default="bestckpt_from_pretrained", help="model ckpt name")                     
    parser.add_argument("--dataset_name", dest="dataset_name", type=str,
                            default="ccdv/cnn_dailymail")
    parser.add_argument("--model", dest="model", type=str,
                        default="PegasusMixPrompt", choices = ["T5Finetune", "T5SoftPrompt", "T5MixPrompt",
                            "BartFinetune", 'BartSoftPrompt', 'BartMixPrompt',
                            "PegasusFinetune", 'PegasusSoftPrompt', 'PegasusMixPrompt', 'CTRLsum', "CTRLsum_origin"])
    parser.add_argument("--model_name", dest="model_name", type=str,
                        default="google/pegasus-large", choices=["t5-base", "google/t5-v1_1-base", "facebook/bart-base",
                        "facebook/bart-large", "google/pegasus-large"])
    parser.add_argument("--use_lm_adapted", dest="use_lm_adapted", type=int,
                        default=0, help="whether to use lm_adapted model") #if we use bart, then automatically don't use lm_adapted
    parser.add_argument("--lm_adapted_path", dest="lm_adapted_path", type=str,
                        default=root + "lm_adapted_t5model/torch_ckpt/large/pytorch_model.bin",
                        help="The path of lm_adapted model")
    parser.add_argument("--cache_path", dest="cache_path", type=str,
                        default=root + "hf_models/pegasus-large/",
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
    # parser.add_argument("--save_model_path", dest="save_model_path", type=str,
    #                         default='/data/ruochen/DATASETS/PromptSumm/xsum/10/seed_0/best_ckpt', help="The path to log dir")
    parser.add_argument("--log_name", dest="log_name", type=str,
                        default='controlling', help="The file name of log file")
    parser.add_argument("--num_workers_summary", dest="num_workers_summary", type=int,
                        default=0, help="dataloader num_workers")
    parser.add_argument("--valid_size_per_gpu_summary", dest="valid_size_per_gpu_summary", type=int,
                        default=128, help="valid size per gpu")
    parser.add_argument("--max_length", dest="max_length", type=int,
                        default=512, help="max sentence length")
    parser.add_argument("--max_guidance_length", dest="max_guidance_length", type=int,
                        default=100)
    parser.add_argument("--local_rank", dest="local_rank", type=int,
                        default=-1, help="local rank")
    parser.add_argument("--few_shot", dest="few_shot", type=str,
                        default='10', help="number of data points for training AND validation")
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
                        default="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/014_c_1070k/bestckpt_full_model", help="path to pretrained model")
    parser.add_argument("--pretrain_prompt_ckpt", type=str,
                        default="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/014_c_1070k/bestckpt_prompt", help="path to pretrained model prompt")
    # parser.add_argument("--big_testset", action='store_true', help="whether or not to evaluate using the 2k testset")  
    parser.add_argument("--full_testset", action='store_true', help="whether or not to evaluate using the full testset")    
    # parser.add_argument("--counterfactual_trained", action='store_true', help="whether or not to use the trained prompt with counterfactuals")  
    parser.add_argument("--seed", dest="seed", type=int,
                        default=42, help="seed for network")
    
    dataset_names = ["ccdv/cnn_dailymail", "xsum", "reddit_tifu", "wikihow", "billsum", "samsum","c4"]
    dataset_versions = ["3.0.0", "default", "long", "all", "default", "samsum",'en']
    text_keys = ["article", "document", "documents", "text", "text", "dialogue"]
    summary_keys = ["highlights", "summary", "tldr", "headline", "summary", "summary"]
    validation_keys = ["validation", "validation", "", "validation", "test", "validation"]
    test_keys = ["test", "test", "", "test", "test", "test"]
    highlights = [True, False, False, False, False, False, False]
    max_summary_lengths = [128, 64, 64, 128, 256, 64]
    
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
    args.dataset_version = dataset_versions[idx]
    args.text_key = text_keys[idx]
    args.summary_key = summary_keys[idx]
    args.validation_key = validation_keys[idx]
    args.test_key = test_keys[idx]
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

def eval_ctrlsum(model, data, ents, idx_to_example, logger, test_ents=None):
    '''
    Args:
        data: list of source texts
        ents: list of control entities, one item for each source text
    '''
    assert len(data) == len(ents)
    model.cuda()
    model.eval()
    # model.half()
    batch_size = args.valid_size_per_gpu_summary

    all_inputs = []
    allypred = []
    all_ents = []

    with torch.no_grad():
        logger.info(f"inf {len(data)} data points")
        for i in range(0, len(data), batch_size): # function as batch iterator
            batch_inputs = data[i:i+batch_size]
            batch_inputs = [v[0] for v in batch_inputs]
            batch_ents = ents[i:i+batch_size]
            
            res = model.sample(batch_inputs, beam=4, prefix_tokens=None, lenpen=1.0, max_len_b=140, min_len=1, no_repeat_ngram_size=3, extra_gen_cls_kwargs=None)
            allypred.extend(res)
            all_ents.extend(batch_ents)
            all_inputs.extend(batch_inputs)
            if i % 20 == 0:
                logger.info(f"inf, {i} data points processed")

    example_success = defaultdict(list) # {example_id: [1,0,...]}, 1 means success
    successes = 0
    success_list = []

    
    for j, (pred, ents) in enumerate(zip(allypred, all_ents)):
        example_id = idx_to_example[j]
        if args.mode == 'single_entity_test':
            condition = ents.lower() in pred.lower()
        elif args.mode == 'k_entity_test':
            # ents = ents.split(' | ')
            ents = test_ents[j]['k_entity']
            condition = all([ent.lower() in pred.lower() for ent in ents])
        elif args.mode == 'oracle':
            cur_oracle_ents = test_ents[j]['oracle']
            condition = all([ent.lower() in pred.lower() for ent in cur_oracle_ents])
        elif args.mode == 'oracle_add_test':
            cur_oracle_ents = test_ents[j]['oracle']
            to_add_ent = test_ents[j]['add']
            condition = all([ent.lower() in pred.lower() for ent in cur_oracle_ents]) and (to_add_ent.lower() in pred.lower())
        success_list.append(condition)
        if condition:
            successes = successes +1
            example_success[example_id].append(1)
        else:
            example_success[example_id].append(0)
    

    return example_success, successes, all_inputs, allypred, all_ents, success_list




def eval(model, valid_dataset, scaler, logger, args, tokenizer, nlp, idx_to_example, seed = 0, test_ents=None):
    def lmap(f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def ids_to_clean_text(generated_ids):
        gen_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return lmap(str.strip, gen_text)

    successes = 0
    model.eval()
    model = model.to(args.device)
    # run and document success number + base number
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = get_dataloader(tokenizer, args.num_workers_summary, valid_dataset, args.valid_size_per_gpu_summary, args.max_length,
                                      args.max_guidance_length, valid_dataset.tokenizer.pad_token_id, valid_sampler, args)
    print('valid_dataloader: ', valid_dataloader)
    all_inputs = []
    allytrue = []
    allypred = []
    all_ents = []
    with torch.no_grad():
        global_batch_step = 0
        for batch in tqdm(valid_dataloader):
            # logger.info(step)
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device),
                      "ents_ids": batch[4].to(args.device), "ents_mask": batch[5].to(args.device)}
            allinp = []
            allent = []
            for k in range(inputs["ents_ids"].shape[0]):
                # print(inputs['input_ids'][k])
                allinp.append(tokenizer.decode(inputs['input_ids'][k], skip_special_tokens=True))
                allent.append(tokenizer.decode(inputs['ents_ids'][k], skip_special_tokens=True))

            if args.model == 'CTRLsum':
                sen = ids_to_clean_text(inputs["input_ids"])
                target = ids_to_clean_text(inputs["target_ids"])
                preds = model.generate(inputs["input_ids"], attention_mask=inputs['attention_mask'], num_beams=4) 
                preds = ids_to_clean_text(preds) 
            else:
                sen, target, preds = model._generative_step(inputs)

            tarres, predres = target, preds
            all_inputs += allinp
            all_ents += allent
            allytrue.extend(tarres)
            allypred.extend(predres)
            global_batch_step += 1

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer = args.stemmer)
    r1s, r2s, rls = [], [], []
    labels = []
    summaries = []
    success_list = []
    example_success = defaultdict(list) # {example_id: [1,0,...]}, 1 means success
    for j in range(len(allytrue)):
        label = allytrue[j]
        summary = allypred[j]
        ents = all_ents[j]
        if args.highlights:
            label = "\n".join(sent_tokenize(label))
            summary = "\n".join(sent_tokenize(summary))
        rouge_score = scorer.score(label, summary)
        labels.append(label)
        summaries.append(summary)
        r1s.append(rouge_score["rouge1"].fmeasure)
        r2s.append(rouge_score["rouge2"].fmeasure)
        rls.append(rouge_score["rougeLsum"].fmeasure)
        # print('ents: ', ents)
        example_id = idx_to_example[j]
        if args.mode == 'single_entity_test':
            condition = ents.lower() in summary.lower()
        elif args.mode == 'k_entity_test':
            ents = ents.split(args.separator)
            # short_summary = ' '.join(summary.split()[:30])
            condition = all([ent.lower() in summary.lower() for ent in ents])
        elif args.mode == 'oracle':
            cur_oracle_ents = test_ents[j]['oracle']
            condition = all([ent.lower() in summary.lower() for ent in cur_oracle_ents])
            # import pdb;pdb.set_trace()
        elif args.mode == 'oracle_add_test':
            cur_oracle_ents = test_ents[j]['oracle']
            to_add_ent = test_ents[j]['add']
            condition = all([ent.lower() in summary.lower() for ent in cur_oracle_ents]) and (to_add_ent.lower() in summary.lower())
        success_list.append(condition)
        if condition:
            successes = successes +1
            example_success[example_id].append(1)
        else:
            example_success[example_id].append(0)
        # entity = ents[0].strip().split(' => ')[0].strip()
        # doc = nlp(summary.strip())
        # if entity in doc.text:
        #     successes += 1
    rouge_score = {
        "rouge1": 100 * np.mean(r1s),
        "rouge2": 100 * np.mean(r2s),
        "rougeLsum": 100 * np.mean(rls)
    }
    p, r, f1 = entity_eval(allytrue, allypred)
    logger.info(f'----Validation Results Summary----')
    logger.info(f'entity-level p: {p}, r: {r}, f1: {f1}')
    logger.info(f'ROUGE score r1: {rouge_score["rouge1"]}, r2: {rouge_score["rouge2"]}, r3: {rouge_score["rougeLsum"]}')
    mean_rouge = (rouge_score["rouge1"] + rouge_score["rouge2"] + rouge_score["rougeLsum"]) / 3
    logger.info(f'mean-rouge: {mean_rouge}')
    logger.info(f'----EVAL FINISHED----')
    return all_inputs, all_ents, labels, summaries, successes, len(allypred), example_success, success_list


def main(args):
    args.filtered = False
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
    # base model
    if args.model == 'CTRLsum_origin':
        model = BARTModel.from_pretrained(
                args.CTRLsum_ckpt_dir,
                checkpoint_file='checkpoint_best.pt',
            ) # no need for tokenizer
        tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/ctrlsum-cnndm",cache_dir=args.cache_path) # no use, just placeholder
        allgentasktokens, answertoken = None, None
        args.few_shot_save_dir = args.data_dir + args.dataset + "/{}/".format(args.few_shot)
    elif args.model == 'CTRLsum':
        model = AutoModelForSeq2SeqLM.from_pretrained("hyunwoongko/ctrlsum-cnndm",cache_dir=args.cache_path)
        tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/ctrlsum-cnndm",cache_dir=args.cache_path)
        allgentasktokens, answertoken = None, None
        args.few_shot_save_dir = args.data_dir + args.dataset + "/{}/".format(args.few_shot)

    else:
        if 'Bart' in args.model:
            basemodel = BartForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
            tokenizer = BartTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)
            args.allnumber_path = 'allnumber.pickle'
        elif 'Pegasus' in args.model:
            basemodel = PegasusForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
            # tokenizer = PegasusTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)
            tokenizer = PegasusTokenizerFast.from_pretrained(args.model_name, cache_dir=args.cache_path)
            logger.info('loaded pegasus models')
            args.allnumber_path = 'allnumber.pickle_newforpegasus'
        else:
            basemodel = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
            tokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)
        model = ModelMixPrompt(args, basemodel, tokenizer, args.model)
        promptnumber = args.prompt_number
        promptembedding = getpromptembedding(model, tokenizer, promptnumber, thistaskname, args.allnumber_path)
        model.set_prompt_embedding(promptnumber, promptembedding)
        # model weights
        if args.use_pretrain_ckpt and  "Finetune" not in args.model:
            print(f"device: {args.device}")
            ckptsum = torch.load(args.pretrain_ckpt, map_location=args.device)
            dicsum = {}
            for x in ckptsum.keys():
                if not (x in ["module.promptnumberforsum", "module.promptembeddingforsum"]):
                    dicsum[x[7:]] = ckptsum[x]
            model.load_state_dict(dicsum)
        logger.info('loaded model')

        args.few_shot_save_dir = args.data_dir + args.dataset + "/{}/".format(args.few_shot)
        
        ## LOAD CKPT
        args.model_save_folder = f'saved_models/{args.dataset}/{args.few_shot}/'
        # if 'MixPrompt' not in args.model:
        #     args.model_save_folder += f'{args.model}/'
        args.model_save_folder += f'{args.model}/'
        args.model_save_path = args.model_save_folder + f'seed_{args.seed}/'
        path = args.model_save_path + args.ckpt_name
        # if args.counterfactual_trained:
        #     path = f'{path}_counterfactual'
        ckptsum = torch.load(path)
        if 'full_weights' in args.ckpt_name:
            model.load_state_dict(ckptsum)
        else:
            model.promptnumber = ckptsum["promptnumber"]
            model.promptembedding = nn.parameter.Parameter(ckptsum["promptembedding"])

        logger.info('loaded ckpt')
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
    # use the whole testset
    valid_file_name = args.data_dir + args.dataset + '/full_test.txt'
    print(valid_file_name)
    args.logger = logger
    # check if we have already generated it
    #if not os.path.isfile(valid_file_name):
    #    dataset_args = [args.dataset_name, args.dataset_version]
    #    subsample_2k_testset(dataset_args, valid_file_name, args.seed, args, n = 1000000, human_eval=True)
    logger.info('generated dataset')
    valid_dataset = SummarizationDataset(valid_file_name, "valid", args.max_length, tokenizer, allgentasktokens, answertoken, args, human_eval = True)
    new_valid_dataset = T5SummarizationDatasetForControlGen(valid_file_name, "valid", args.max_length, tokenizer, allgentasktokens, answertoken, args)
        
    valid_dataset.data = valid_dataset.data[:100]
    scaler = None
    # find all entities for all inputs
    # lead 3 VS others
    lead3_ents = []
    other_ents = []
    full_ents = []
    oracle_ents = []
    for i, (inp, tar) in tqdm(enumerate(valid_dataset.data)):
        # get oracle entities
        tgt_sents = ''.join(sent_tokenize(tar))
        oraclents = valid_dataset.spacy_nlp(tgt_sents).ents
        oraclents = [ent.text for ent in oraclents]
        oracle_ents.append(oraclents)
        # get src entities
        inp = ' '.join(inp.split()[:400]) # mimic truncation
        sents = sent_tokenize(inp)
        if len(sents)>3:
            lead3 = ''.join(sents[:3])
            other = ''.join(sents[3:])
            full = ''.join(sents)
        else:
            lead3 = inp
            other = ''
            full = ''.join(sents)
        lead3ents = valid_dataset.spacy_nlp(lead3).ents
        lead3ents = [ent.text for ent in lead3ents]
        lead3_ents.append(lead3ents)
        otherents = valid_dataset.spacy_nlp(other).ents
        otherents = [ent.text for ent in otherents]
        other_ents.append(otherents)
        fullents = valid_dataset.spacy_nlp(full).ents
        fullents = [ent.text for ent in fullents]
        full_ents.append(fullents)
    names = ['FULL', 'LEAD3', 'OTHER']

    nlp = English()
    # sentencizer = nlp.create_pipe("")
    nlp.add_pipe('sentencizer')
    if args.mode == 'single_entity_test':
        for (idx, all_ents) in enumerate([full_ents, lead3_ents, other_ents]):
            logger.info(f'-------------{names[idx]}---------------')
            # max number of entities
            number_of_ents = [len(i) for i in all_ents]
            max_number = max(number_of_ents)
            logger.info(f'max number: {max_number}')
            # max_number = 1 #comment this out except for testing
            # loop at each entity number, up to max
            allsec = 0
            allnum = 0
            # for i in range(max_number):
            # logger.info(f'number {i}')
            # create new sub-validset
            # valid_indices = [j for (j, n) in enumerate(number_of_ents) if n>i]
            
            # new_valid_dataset = copy.deepcopy(valid_dataset)
            new_data = []
            new_allents = []
            idx = 0
            idx_to_example = {} # idx of eval case : example id
            for vi in range(valid_dataset.num_entries):
                original_data = valid_dataset.data[vi]
                original_ents = all_ents[vi]
                for ent in set(original_ents):
                    # filter non-standard entities (spacy entities are noisy)
                    if '.' in ent:
                        continue
                    if args.single_word and len(ent.split()) > 1:
                        continue
                    if "CTRLsum" in args.model:
                        new_text = f'{str(ent)} => {original_data[0]}'
                        new_data.append([new_text, original_data[1]])
                    else:
                        new_data.append(original_data)
                    new_allents.append(args.separator.join([str(ent)]))
                    idx_to_example[idx] = vi
                    idx += 1

            if args.model == 'CTRLsum_origin':
                logger.info(f'start CTRLsum_origin eval')
                example_successes, successes, inputs0, summaries0, all_ents0 = eval_ctrlsum(model, new_data, new_allents, idx_to_example, logger)
                allsec += successes
                allnum += len(new_data)
            else:
                new_valid_dataset.data = new_data
                new_valid_dataset.allent = new_allents
                new_valid_dataset.num_entries = len(new_data)
                inputs0, all_ents0, labels0, summaries0, successes, num, example_successes = eval(model, new_valid_dataset, scaler, logger, args, tokenizer, nlp, idx_to_example)
                allsec += successes
                allnum += num

            for i in range(min(50, len(inputs0))): # only print limited examples
                logger.info('-----')
                logger.info(f'INPUT: {inputs0[i]}')
                # logger.info(f'LABEL: {labels0[i]}')
                logger.info(f'ENTS: {all_ents0[i]}; SUMMARY: {summaries0[i]}')

            logger.info(f'successes: {allsec}; number: {allnum}')
            logger.info(f'success rate: {allsec/allnum}')

            example_success_rates = [np.mean(v) for v in example_successes.values()]
            logger.info(f"success by examples, examples #{len(example_success_rates)}, mean: {np.mean(example_success_rates)}, \
                        min: {np.min(example_success_rates)}, max: {np.max(example_success_rates)}, std: {np.std(example_success_rates)}")
    elif args.mode == 'k_entity_test':
        test_k_entity = int(args.k_entity)
        for (idx, all_ents) in enumerate([full_ents]):
            logger.info(f'-------------{names[idx]}---------------')
            # max number of entities
            number_of_ents = [len(i) for i in all_ents]
            max_number = max(number_of_ents)
            logger.info(f'max number: {max_number}')
            # max_number = 1 #comment this out except for testing
            # loop at each entity number, up to max
            allsec = 0
            allnum = 0
            # for i in range(max_number):
            # logger.info(f'number {i}')
            # create new sub-validset
            # valid_indices = [j for (j, n) in enumerate(number_of_ents) if n>i]
            
            # new_valid_dataset = copy.deepcopy(valid_dataset)
            new_data = []
            new_allents = []
            idx = 0
            idx_to_example = {} # idx of eval case : example id
            ents_to_pass = {}
            for vi in range(valid_dataset.num_entries):
                # if vi >= 20:
                #     break
                original_data = valid_dataset.data[vi]
                original_ents = all_ents[vi]
                original_ents = list(set([ent for ent in original_ents if '.' not in ent]))
                # sample max 40 unique tuple of entities
                sample_iter, sampled_ent_tuples = 0, set()
                if len(original_ents) < test_k_entity:
                    logger.info(f"too few entities, skip")
                    continue
                while len(sampled_ent_tuples) < 20 and sample_iter < 60:
                    ents = tuple(random.sample(original_ents, test_k_entity))
                    if ents not in sampled_ent_tuples:
                        sampled_ent_tuples.add(ents)
                    sample_iter += 1
                
                for cur_ents in sampled_ent_tuples:
                    if "CTRLsum" in args.model:
                        ent_str = ' | '.join(cur_ents)
                        # ent_str = ' '.join(cur_ents)
                        new_text = f'{ent_str} => {original_data[0]}'
                        new_data.append([new_text, original_data[1]])
                        new_allents.append(ent_str)
                    else:
                        ent_str = args.separator.join(cur_ents)
                        new_data.append(original_data)
                        new_allents.append(ent_str)
                    ents_to_pass[idx] = {'k_entity':cur_ents}
                    idx_to_example[idx] = vi
                    idx += 1
            logger.info(f"all data length: {len(new_data)}")

            if args.model == 'CTRLsum_origin':
                logger.info(f'start CTRLsum_origin eval')
                example_successes, successes, inputs0, summaries0, all_ents0, suc_list = eval_ctrlsum(model, new_data, new_allents, idx_to_example, logger, test_ents=ents_to_pass)
                allsec += successes
                allnum += len(new_data)
            else:
                logger.info(f'start eval')
                new_valid_dataset.data = new_data
                new_valid_dataset.allent = new_allents
                new_valid_dataset.num_entries = len(new_data)
                inputs0, all_ents0, labels0, summaries0, successes, num, example_successes, suc_list = eval(model, new_valid_dataset, scaler, logger, args, tokenizer, nlp, idx_to_example)
                allsec += successes
                allnum += num

            example_id = 0
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer = args.stemmer)
            for i in range(min(1000, len(inputs0))): # only print limited examples
                cur_example_id = idx_to_example[i]
                if cur_example_id > example_id: # new example
                    example_id = cur_example_id
                    logger.info('---------------------------')
                    logger.info("{} INPUT: {}".format(example_id, inputs0[i]))
                    logger.info("LABEL: {}".format(labels0[i].replace("\n", " ")))
                if suc_list[i]:
                    label = "\n".join(sent_tokenize(labels0[i]))
                    summary = " \n".join(sent_tokenize(summaries0[i]))
                    rouge_score = scorer.score(label, summary)
                    r1 = 100 * rouge_score["rouge1"].fmeasure
                    r2 = 100 * rouge_score["rouge2"].fmeasure
                    rl = 100 * rouge_score["rougeLsum"].fmeasure
                    mean_r = (r1 + r2 + rl) / 3
                    logger.info("ENTS: {}; success: {}, mean-R: {:.4f} \nSUMMARY: {}".format(all_ents0[i], suc_list[i], mean_r, summaries0[i].replace("\n", " ")))
            print("all lengths", len(inputs0), len(labels0), len(all_ents0), len(summaries0))

            logger.info(f'successes: {allsec}; number: {allnum}')
            logger.info(f'success rate: {allsec/allnum}')

            example_success_rates = [np.mean(v) for v in example_successes.values()]
            logger.info(f"success by examples, examples #{len(example_success_rates)}, mean: {np.mean(example_success_rates)}, \
                        min: {np.min(example_success_rates)}, max: {np.max(example_success_rates)}, std: {np.std(example_success_rates)}")
    elif args.mode == 'oracle':
        allsec = 0
        allnum = 0
        new_data = []
        new_allents = []
        ents_to_pass = {}
        idx = 0
        idx_to_example = {} # idx of eval case : example id
        for vi in range(valid_dataset.num_entries):
            original_data = valid_dataset.data[vi]
            cur_oracle_ents = oracle_ents[vi]
            if len(cur_oracle_ents) == 0: # no entity, skip this example
                continue
            ents_to_pass[idx] = {'oracle':cur_oracle_ents}
            if "CTRLsum" in args.model:
                ent_str = ' | '.join(cur_oracle_ents)
                # ent_str = ' '.join(cur_oracle_ents)  
                new_text = f'{ent_str} => {original_data[0]}'
                new_data.append([new_text, original_data[1]])
                new_allents.append(ent_str)
            else:
                ent_str = args.separator.join(cur_oracle_ents)
                new_data.append(original_data)
                new_allents.append(ent_str)
            idx_to_example[idx] = vi
            idx += 1
        
        if args.model == 'CTRLsum_origin':
            logger.info(f'start CTRLsum_origin eval')
            example_successes, successes, inputs0, summaries0, all_ents0, suc_list = eval_ctrlsum(model, new_data, new_allents, idx_to_example, logger, test_ents=ents_to_pass)
            allsec += successes
            allnum += len(new_data)
        else:
            new_valid_dataset.data = new_data
            new_valid_dataset.allent = new_allents
            new_valid_dataset.num_entries = len(new_data)
            inputs0, all_ents0, labels0, summaries0, successes, num, example_successes, suc_list = eval(model, new_valid_dataset, scaler, logger, args, tokenizer, nlp, idx_to_example, test_ents=ents_to_pass)
            allsec += successes
            allnum += num

        for i in range(min(1000, len(inputs0))): # only print limited examples
            logger.info('-----')
            logger.info(f'INPUT: {inputs0[i]}')
            # logger.info(f'LABEL: {labels0[i]}')
            logger.info(f'ENTS: {all_ents0[i]}; success: {suc_list[i]}; SUMMARY: {summaries0[i]}')

        logger.info(f'successes: {allsec}; number: {allnum}')
        logger.info(f'success rate: {allsec/allnum}')

        example_success_rates = [np.mean(v) for v in example_successes.values()]
        logger.info(f"success by examples, examples #{len(example_success_rates)}, mean: {np.mean(example_success_rates)}, \
                    min: {np.min(example_success_rates)}, max: {np.max(example_success_rates)}, std: {np.std(example_success_rates)}")
    elif args.mode == 'interactive':
        model.eval()
        model = model.to(args.device)
        username = input("What is your name? ")
        name_list = [i for i in os.listdir('../human_evaluation/users/controllable_{}/'.format(args.dataset))]
        if not(username in name_list):
            os.mkdir('../human_evaluation/users/controllable_{}/{}/'.format(args.dataset, username))
        n_attempts = 3
        count = 0
        with torch.no_grad():
            while True:
                # interactive loop
                index = int(input("select index: "))
                inp, tar = valid_dataset.data[index]

                logger.info("\n" + "*"*50 + "\n")
                logger.info("\nSelected article: {}".format(index))
                logger.info("\nSource document: \n{}".format(inp[:2000]))
                #logger.info("Ground truth summary: \n{}".format(tar))

                if True: # show entities
                    _inp = ' '.join(inp.split())
                    _inp_ents = valid_dataset.spacy_nlp(_inp).ents
                    _inp_ents_text = set([ent.text for ent in _inp_ents])
                    logger.info("\nSource entities: \n{}".format(_inp_ents_text))
                logger.info("\nSuggested entity chain length: CNN/DM: 6, Xsum: 3, BillSum: 14, SAMSum: 3")
    
                # process data
                input_res = valid_dataset.tokenizer.batch_encode_plus([inp], padding=False, max_length=valid_dataset.maxlen, truncation=True, return_tensors="pt")
                input_ids = input_res['input_ids']
                input_attn_mask = input_res['attention_mask']

                target_res = valid_dataset.tokenizer.batch_encode_plus([tar], padding=False, max_length=valid_dataset.maxlen, truncation=True, return_tensors="pt")
                target_ids = target_res['input_ids']
                target_attn_mask = target_res['attention_mask']

                count = 0
                satisfied = False
                ents, summaries, satis, causes = [], [], [], []
                while count < n_attempts and not(satisfied):
                    logger.info("\nAttempt {} / {}".format(count + 1, n_attempts))
                    ent_chain = input("\nInput entity chain (Separate entities with commas, no space, no quotation mark, like: entity_1,entity_2): ")
                    valid_chain = len([x for x in ent_chain.split(",") if x in _inp_ents_text]) == len(ent_chain.split(","))
                    while not(valid_chain):
                        ent_chain = input("\nInput entity chain (Separate entities with commas, no space, no quotation mark, like: entity_1,entity_2): ")
                        valid_chain = len([x for x in ent_chain.split(",") if x in _inp_ents_text]) == len(ent_chain.split(","))

                    ents.append(ent_chain)
                    ent_res = valid_dataset.tokenizer.batch_encode_plus([ent_chain], padding=False, max_length=valid_dataset.maxlen, truncation=True, return_tensors="pt")
                    ent_ids = ent_res['input_ids']
                    ent_attn_mask = ent_res['attention_mask']

                    inputs = {"input_ids": input_ids.to(args.device), "attention_mask":input_attn_mask.to(args.device),
                                "target_ids": target_ids.to(args.device), "target_mask": target_attn_mask.to(args.device),
                                "ents_ids": ent_ids.to(args.device), "ents_mask": ent_attn_mask.to(args.device)}

                    if "CTRLsum" in args.model:
                        new_text = f'{ent_chain} => {inp}'
                        preds = model.sample([new_text], beam=4, prefix_tokens=None, lenpen=1.0, max_len_b=140, min_len=1, no_repeat_ngram_size=3, extra_gen_cls_kwargs=None)
                    else:
                        sen, target, preds = model._generative_step(inputs)

                    summary = preds[0]
                    logger.info("Generated summary: \n{}".format(summary))
                    summaries.append(summary)

                    count += 1
                    satisfied = input("Are you satisfied with the summary? (Yes/No answer) ") == "Yes"
                    satis.append(satisfied)

                    cause = "_"
                    if satisfied == False:
                        cause = input("What is wrong? Pick a number among (1) Not grammatical / (2) Not factual / (3) Not containing the entities ")
                        causes.append(cause)
                if satisfied:
                    logger.info("Reached a good summary in {} attempts".format(count))
                else:
                    logger.info("Did not reach a good summary")

                dic = {
                    "source": inp,
                    "target": tar,
                    "entity chains": ents,
                    "generated summaries": summaries,
                    "satisfaction": satis,
                    "causes": causes
                }
                save_path = '../human_evaluation/users/controllable_{}/{}/{}.json'.format(args.dataset, username, index)
                with open(save_path, 'w') as outfile:
                    json.dump(dic, outfile)
                    print("saved the results!", save_path)
            
    elif args.mode == 'oracle_add_test':
        for (idx, all_ents) in enumerate([full_ents, lead3_ents, other_ents]):
            logger.info(f'-------------{names[idx]}---------------')
            allsec = 0
            allnum = 0
            new_data = []
            new_allents = []
            ents_to_pass = {}
            idx = 0
            idx_to_example = {} # idx of eval case : example id
            for vi in range(valid_dataset.num_entries):
                original_data = valid_dataset.data[vi]
                original_ents = all_ents[vi]
                cur_oracle_ents = oracle_ents[vi]
                for ent in (set(original_ents)-set(cur_oracle_ents)):
                    # filter non-standard entities (spacy entities are noisy)
                    if '.' in ent:
                        continue
                    if args.single_word and len(ent.split()) > 1:
                        continue
                    new_ents = cur_oracle_ents + [ent]
                    ents_to_pass[idx] = {'oracle':cur_oracle_ents, 'add':ent}
                    if "CTRLsum" in args.model:
                        ent_str = ' | '.join(new_ents)
                        new_text = f'{ent_str} => {original_data[0]}'
                        new_data.append([new_text, original_data[1]])
                        new_allents.append(ent_str)
                    else:
                        ent_str = args.separator.join(new_ents)
                        new_data.append(original_data)
                        new_allents.append(ent_str)
                    
                    idx_to_example[idx] = vi
                    idx += 1
            
            if args.model == 'CTRLsum_origin':
                logger.info(f'start CTRLsum_origin eval')
                example_successes, successes, inputs0, summaries0, all_ents0 = eval_ctrlsum(model, new_data, new_allents, idx_to_example, logger, test_ents=ents_to_pass)
                allsec += successes
                allnum += len(new_data)
            else:
                new_valid_dataset.data = new_data
                new_valid_dataset.allent = new_allents
                new_valid_dataset.num_entries = len(new_data)
                inputs0, all_ents0, labels0, summaries0, successes, num, example_successes = eval(model, new_valid_dataset, scaler, logger, args, tokenizer, nlp, idx_to_example, test_ents=ents_to_pass)
                allsec += successes
                allnum += num
            logger.info(f'successes: {allsec}; number: {allnum}')
            logger.info(f'success rate: {allsec/allnum}')

            example_success_rates = [np.mean(v) for v in example_successes.values()]
            logger.info(f"success by examples, examples #{len(example_success_rates)}, mean: {np.mean(example_success_rates)}, \
                        min: {np.min(example_success_rates)}, max: {np.max(example_success_rates)}, std: {np.std(example_success_rates)}")

    elif args.mode == 'oracle_drop_test':
        pass
if __name__ == "__main__":
    args = set_args()
    set_logger(args)
    logger.info(args)
    seed_everything(args)
    main(args)


