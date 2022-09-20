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
import random
import copy
from dataset import *
from nltk.tokenize import sent_tokenize
from spacy.lang.en import English

def set_args():
    parser = argparse.ArgumentParser(description="latentRE")
    data_root = "/data/ruochen/"
    root = "/data/mathieu/"
    parser.add_argument("--data_dir", dest="data_dir", type=str,
                        default= data_root + "DATASETS/PromptSumm/")
    parser.add_argument("--dataset_name", dest="dataset_name", type=str,
                            default="ccdv/cnn_dailymail")
    parser.add_argument("--model", dest="model", type=str,
                        default="PegasusMixPrompt", choices = ["T5Finetune", "T5SoftPrompt", "T5MixPrompt",
                            "BartFinetune", 'BartSoftPrompt', 'BartMixPrompt',
                            "PegasusFinetune", 'PegasusSoftPrompt', 'PegasusMixPrompt'])
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

def eval(model, valid_dataset, scaler, logger, args, tokenizer, nlp, seed = 0):
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
        for batch in valid_dataloader:
            # logger.info(step)
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device),
                      "ents_ids": batch[4].to(args.device), "ents_mask": batch[5].to(args.device)}
            allinp = []
            allent = []
            for k in range(inputs["ents_ids"].shape[0]):
                # print(inputs['input_ids'][k])
                allinp.append(tokenizer.decode(inputs['input_ids'][k]))
                allent.append(tokenizer.decode(inputs['ents_ids'][k]))
            if scaler is not None:
                with autocast():
                    sen, target, preds = model._generative_step(inputs)
                    tarres, predres = target, preds
            else:
                sen, target, preds = model._generative_step(inputs)
                tarres, predres = target, preds
            all_inputs += allinp
            all_ents += allent
            allytrue.extend(tarres)
            allypred.extend(predres)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer = args.stemmer)
    r1s, r2s, rls = [], [], []
    labels = []
    summaries = []
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
        print('ents: ', ents)
        if ents[0].lower() in summary.lower():
            successes = successes +1
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
    return all_inputs, all_ents, labels, summaries, successes, len(allypred)


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
    if 'Bart' in args.model:
        basemodel = BartForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
        tokenizer = BartTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)
    elif 'Pegasus' in args.model:
        basemodel = PegasusForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
        tokenizer = PegasusTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)
        logger.info('loaded pegasus models')
    else:
        basemodel = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
        tokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)
    model = ModelMixPrompt(args, basemodel, tokenizer, args.model)
    promptnumber = args.prompt_number
    promptembedding = getpromptembedding(model, tokenizer, promptnumber, thistaskname)
    model.set_prompt_embedding(promptnumber, promptembedding)
    # model weights
    if args.use_pretrain_ckpt and  "Finetune" not in args.model:
        ckptsum = torch.load(args.pretrain_ckpt, map_location="cuda:0")
        dicsum = {}
        for x in ckptsum.keys():
            if not (x in ["module.promptnumberforsum", "module.promptembeddingforsum"]):
                dicsum[x[7:]] = ckptsum[x]
        model.load_state_dict(dicsum)
    logger.info('loaded model')
    seed = 0
    args.few_shot_save_dir = args.data_dir + args.dataset + "/{}/".format(args.few_shot)
    
    ## LOAD CKPT
    args.model_save_folder = f'saved_models/{args.dataset}/{args.few_shot}/'
    if 'MixPrompt' not in args.model:
        args.model_save_folder += f'{args.model}/'
    args.model_save_path = args.model_save_folder + f'seed_{seed}/'
    path = args.model_save_path + 'bestckpt'
    # if args.counterfactual_trained:
    #     path = f'{path}_counterfactual'
    ckptsum = torch.load(path)
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
    valid_file_name = args.data_dir + args.dataset + '/100_test.txt'
    args.logger = logger
    # check if we have already generated it
    if not os.path.isfile(valid_file_name):
        dataset_args = [args.dataset_name, args.dataset_version]
        subsample_2k_testset(dataset_args, valid_file_name, args.seed, args, n = 100)
    logger.info('generated dataset')
    valid_dataset = T5SummarizationDataset(valid_file_name, "valid", args.max_length, tokenizer, allgentasktokens, answertoken, args)
        
    scaler = None
    # find all entities for all inputs
    # lead 3 VS others
    lead3_ents = []
    other_ents = []
    for i, (inp, tar) in enumerate(valid_dataset.data):
        sents = sent_tokenize(inp)
        if len(sents)>3:
            lead3 = ''.join(sents[:3])
            other = ''.join(sents[3:])
        else:
            lead3 = inp
            other = ''
        lead3ents = valid_dataset.spacy_nlp(lead3).ents
        lead3ents = [ent.text for ent in lead3ents]
        lead3_ents.append(lead3ents)
        otherents = valid_dataset.spacy_nlp(other).ents
        otherents = [ent.text for ent in otherents]
        other_ents.append(otherents)
    names = ['LEAD3', 'OTHER']

    nlp = English()
    # sentencizer = nlp.create_pipe("")
    nlp.add_pipe('sentencizer')
    for (idx, all_ents) in enumerate([lead3_ents, other_ents]):
        logger.info(f'-------------{names[idx]}---------------')
        # max number of entities
        number_of_ents = [len(i) for i in all_ents]
        max_number = max(number_of_ents)
        logger.info(f'max number: {max_number}')
        # max_number = 1 #comment this out except for testing
        # loop at each entity number, up to max
        allsec = 0
        allnum = 0
        for i in range(max_number):
            logger.info(f'number {i}')
            # create new sub-validset
            valid_indices = [j for (j, n) in enumerate(number_of_ents) if n>i]
            new_valid_dataset = copy.deepcopy(valid_dataset)
            new_data = []
            new_allents = {}
            for vi in valid_indices:
                original_data = valid_dataset.data[vi]
                new_data.append(original_data)
                new_allents[re.sub(' +', ' ', original_data[0])] = args.separator.join([all_ents[vi][i]])
            new_valid_dataset.data = new_data
            new_valid_dataset.allent = new_allents
            new_valid_dataset.num_entries = len(new_data)
            inputs0, all_ents0, labels0, summaries0, successes, num = eval(model, new_valid_dataset, scaler, logger, args, tokenizer, nlp)
            allsec += successes
            allnum += num
        for i in range(min(10, len(inputs0))): # only print limited examples
            logger.info('-----')
            logger.info(f'INPUT: {inputs0[i]}')
            logger.info(f'LABEL: {labels0[i]}')
            logger.info(f'ENTS: {all_ents0[i]}; SUMMARY: {summaries0[i]}')

        logger.info(f'successes: {allsec}; number: {allnum}')
        logger.info(f'success rate: {allsec/allnum}')
if __name__ == "__main__":
    args = set_args()
    set_logger(args)
    logger.info(args)
    main(args)


