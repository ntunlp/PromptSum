# This script performs qualitative tests to test a model for controllablity
import argparse
import copy
from nltk.tokenize import sent_tokenize
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, PegasusConfig, PegasusTokenizerFast

from dataset_entity import *
from dataset_summary import *
from engine_pretrain import *
from engine_entity import *
from engine_summary import *
from models.model_summary_mix import *



def set_args():
    parser = argparse.ArgumentParser(description="latentRE")
    root = "/export/home/"
    data_root = "/export/home/"
    parser.add_argument("--data_dir", dest="data_dir", type=str,
                        default= data_root + "dataset/PromptSumm/")
    parser.add_argument("--dataset_name", dest="dataset_name", type=str,
                            default="ccdv/cnn_dailymail")
    parser.add_argument("--model", dest="model", type=str,
                        default="PegasusMixPrompt", choices = ["T5Finetune", "T5SoftPrompt", "T5MixPrompt",
                            "BartFinetune", 'BartSoftPrompt', 'BartMixPrompt',
                            "PegasusFinetune", 'PegasusSoftPrompt', 'PegasusMixPrompt', 'CTRLsum', "CTRLsum_origin"])
    parser.add_argument("--ckpt_name", dest="ckpt_name", type=str,
                        default="bestckpt_from_pretrained", help="model ckpt name") 
    parser.add_argument("--tagger_ckpt_name", dest="tagger_ckpt_name", type=str,
                        default="bestckpt_prompt", help="tagger ckpt name")                     
    parser.add_argument("--model_name", dest="model_name", type=str,
                        default="google/pegasus-large", choices=["t5-base", "google/t5-v1_1-base", "facebook/bart-base",
                        "facebook/bart-large", "google/pegasus-large"])
    parser.add_argument("--use_lm_adapted", dest="use_lm_adapted", type=int,
                        default=1, help="whether to use lm_adapted model") #if we use bart, then automatically don't use lm_adapted
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
                        default=16, help="valid size per gpu")
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
    parser.add_argument("--full_testset", action='store_true', help="whether or not to evaluate using the full testset")
    parser.add_argument("--counterfactual_trained", action='store_true', help="whether or not to use the trained prompt with counterfactuals")  
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
    args.max_position_embeddings = 1024
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

def hallucinated(entity_chain, input, spacy_nlp):
    '''
    Returns: 1. non-hallucinated entity chains with hallucination removed
            2. boolean on whether it was hallucinated or not
    '''
    input_guidance_list = entity_chain.split(',')
    ents_true = spacy_nlp(input).ents
    ents_true = [ent.text for ent in ents_true]
    hallucinated = [ent for ent in input_guidance_list if ent not in ents_true]
    # hallucinated = [ent for ent in input_guidance_list if ent.lower() not in input.lower()]
    if len(hallucinated)>0:
        non_hallucinated = [ent for ent in input_guidance_list if ent not in hallucinated]
        return ','.join(non_hallucinated), True
    else:
        return ','.join(input_guidance_list), False

def count_hallucination_percentage(allytrue, allypred, spacy_nlp): #CAN CHANGE HERE
    # hallucinated_count = 0
    # all_count = 0
    hallucination_rates = []
    no_entity_preds = 0
    for i, ytrue in enumerate(allytrue):
        ents_true = spacy_nlp(ytrue).ents
        ents_true = [ent.text for ent in ents_true]
        ypred = allypred[i]
        ents_pred = spacy_nlp(ypred).ents
        ents_pred = [ent.text for ent in ents_pred]
        hallucinated_ents = [ent for ent in ents_pred if ent not in ents_true]
        # hallucinated_count += len(hallucinated_ents)
        # all_count += len(ents_pred)
        if len(ents_pred) == 0:
            no_entity_preds += 1
            # logger.info(f"no ent predicted: {ypred}")
        else:
            hallucination_rates.append(len(hallucinated_ents)/len(ents_pred))
    logger.info(f"in count_hallucination_percentage: # no entity preds: {no_entity_preds}")
    return hallucination_rates

def new_valid_dataset(valid_dataset, indices, ents):
    # new_valid_dataset = T5SummarizationDatasetForControlGen('dummy', "valid", args.max_length, tokenizer, allgentasktokens, answertoken, args)

    valid_dataset_new = copy.deepcopy(valid_dataset)
    valid_dataset_new.data =[valid_dataset_new.data[i] for i in indices]
    valid_dataset_new.num_entries = len(valid_dataset_new.data)
    valid_dataset_new.allent = ents
    return valid_dataset_new

def eval(model, valid_dataset, scaler, logger, args, tokenizer, spacy_nlp, seed = 0):
    model.eval()
    model = model.to(args.device)
    all_ents = []
    if args.use_t5_tagger and args.model in ['T5MixPrompt', 'PegasusMixPrompt'] and args.guidance_mode != "target":
        if args.infer_val_entities:
            ########## predict the validation entity chains with the 1st prompt tuning stage model
            # entbasemodel = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir = args.cache_path)
            # enttokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir = args.cache_path)
            # entmodel = T5forFinetuneEntity(entbasemodel, enttokenizer, args)
            entbasemodel = PegasusForConditionalGeneration.from_pretrained(args.model_name, max_position_embeddings = args.max_position_embeddings, cache_dir = args.cache_path)
            enttokenizer = PegasusTokenizerFast.from_pretrained(args.model_name, cache_dir = args.cache_path)
            entmodel = ModelEntity(entbasemodel, enttokenizer, args)
            logger.info("Loading the pre-trained NER model!")

            # model weights
            ckpt = torch.load(args.pretrain_ckpt, map_location="cuda:0")
            dic = {}
            for x in ckpt.keys():
                if not (x in ["module.promptnumber", "module.promptembedding", "module.promptnumberforsum", "module.promptembeddingforsum"]):
                    dic[x[7:]] = ckpt[x]
            entmodel.load_state_dict(dic)

            # just prompt
            #onepath = f'{args.few_shot_save_dir}seed_{seed}/data_for_bert_{seed}/tagger/bestckpt_prompt' ####bestckpt_prompt?
            onepath = f'entity_ckpt/{args.dataset}/{args.few_shot}/seed_{seed}/{args.tagger_ckpt_name}'
            print(onepath)
            oneckpt = torch.load(onepath, map_location="cuda:0")
            entmodel.promptnumber = oneckpt["promptnumber"]
            entmodel.promptembedding = oneckpt["promptembedding"]
        
            n_params = sum(p.numel() for p in entmodel.parameters() if p.requires_grad)
            logger.info("The ent model has {} trainable parameters".format(n_params))
            entmodel.to(args.device)
            logger.info("move to device!")
            model.eval()

            respath = f'entity_ckpt/{args.dataset}/{args.few_shot}/seed_{seed}/T5valident.pkl'
            logger.info(f'respath: {respath}')
            if not os.path.isfile(respath):
                logger.info('generating')
                #XSUM INFER
                alldata = valid_dataset.data
                allresofvalid = {}
                all_ents = []
                with torch.no_grad():
                    for step in range(len(alldata)):
                        if step % 100 == 0:
                            logger.info(f"gen step: {step}")
                        onedata = alldata[step]
                        inputdata = onedata[0]
                        tempdata = re.sub(' +', ' ', inputdata)
                        inputres = enttokenizer.batch_encode_plus([tempdata], padding=True, max_length=args.max_length, truncation=True, return_tensors="pt")
                        input_ids = inputres["input_ids"].to(args.device)
                        attention_mask = inputres["attention_mask"].to(args.device)
                        input = {"input_ids": input_ids, "attention_mask": attention_mask}
                        _, _, tagpreds = entmodel._generative_step(input)
                        allentitylist = tagpreds[0].split(',')
                        if allentitylist == []:
                            input_guidance = 'none'
                        else:
                            input_guidance = args.separator.join(list(dict.fromkeys(allentitylist)))
                        allresofvalid[tempdata] = input_guidance
                        all_ents.append(input_guidance)
                logger.info(len(allresofvalid))
                with open(respath, "wb") as f:
                    pickle.dump(allresofvalid, f)
                    logger.info("saved the T5 valid entities")
                torch.cuda.empty_cache()
                del entmodel, enttokenizer
                gc.collect()
            else:
                # we already have it
                with open(respath, "rb") as f:
                    allresofvalid = pickle.load(f)
                all_ents = list(allresofvalid.values())
    valid_dataset.allent = allresofvalid
    # split into nh and h datasets
    dic = {'nh_indices': [], 'nh_ents': {},
             'h_indices': [], 'h_ents': {}, 'hr_ents': {}}
    for i, data in enumerate(valid_dataset.data):
        inputdata = data[0]
        tempdata = re.sub(' +', ' ', inputdata)
        try:
            entity_chain = valid_dataset.allent[tempdata]
        except:
            logger.info(f'DATA: {data}')
            logger.info(f'cannot find :{tempdata}')
            logger.info(f'dict :{valid_dataset.allent}')
            raise Exception('end')
        non_halents, hal = hallucinated(entity_chain, inputdata, spacy_nlp)
        if hal:
            dic['h_indices'].append(i)
            dic['h_ents'][tempdata] = entity_chain
            dic['hr_ents'][tempdata] = non_halents
        else:
            dic['nh_indices'].append(i)
            dic['nh_ents'][tempdata] = entity_chain
        
    valid_dataset_nh = new_valid_dataset(valid_dataset, dic['nh_indices'], dic['nh_ents'])
    valid_dataset_h = new_valid_dataset(valid_dataset, dic['h_indices'], dic['h_ents'])
    valid_dataset_hr = new_valid_dataset(valid_dataset, dic['h_indices'], dic['hr_ents'])
    
    logger.info(f'{valid_dataset_h.num_entries} hallucinated examples')
    logger.info(f'{valid_dataset_nh.num_entries} NON-hallucinated examples')
    names = ['NOT HALLUCINATED', 'HALLUCINATED AS-IS', 'HALLUCINATED WITH HALLUCINATION REMOVED']
    for idx, valid_dataset in enumerate([valid_dataset_nh, valid_dataset_h, valid_dataset_hr]):
        valid_sampler = SequentialSampler(valid_dataset)
        valid_dataloader = get_dataloader(tokenizer, args.num_workers_summary, valid_dataset, args.valid_size_per_gpu_summary, args.max_length,
                                        args.max_guidance_length, valid_dataset.tokenizer.pad_token_id, valid_sampler, args)
        all_inputs = []
        allytrue = []
        allypred = []
        allent = []
        with torch.no_grad():
            for step, batch in enumerate(valid_dataloader):
                # logger.info(step)
                inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device),
                      "ents_ids": batch[4].to(args.device), "ents_mask": batch[5].to(args.device)}
                
                allinp = []
                for k in range(inputs["ents_ids"].shape[0]):
                    # print(inputs['input_ids'][k])
                    allinp.append(tokenizer.decode(inputs['input_ids'][k], skip_special_tokens=True))
                    allent.append(tokenizer.decode(inputs['ents_ids'][k], skip_special_tokens=True))

                if scaler is not None:
                    with autocast():
                        sen, target, preds = model._generative_step(inputs)
                        tarres, predres = target, preds
                else:
                    sen, target, preds = model._generative_step(inputs)
                    tarres, predres = target, preds
                all_inputs += allinp
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
        p, r, f1 = entity_eval(allytrue, allypred)
        # percentage of hallucinated entities
        percent_hals = count_hallucination_percentage(all_inputs, allypred, spacy_nlp)

        for i in range(min(10, len(allypred))): # only print limited examples
            logger.info('-----')
            logger.info(f'INPUT: {all_inputs[i]}')
            # logger.info(f'LABEL: {labels0[i]}')
            logger.info(f'ENTS: {allent[i]}; hallucination rate: {percent_hals[i]}; SUMMARY: {allypred[i]}')

        logger.info(f'--------{names[idx]}--------')
        logger.info(f'entity-level p: {p}, r: {r}, f1: {f1}')
        logger.info(f'ROUGE score r1: {rouge_score["rouge1"]}, r2: {rouge_score["rouge2"]}, r3: {rouge_score["rougeLsum"]}')
        mean_rouge = (rouge_score["rouge1"] + rouge_score["rouge2"] + rouge_score["rougeLsum"]) / 3
        logger.info(f'mean-rouge: {mean_rouge}')
        logger.info(f'Percentage of hallucination in predictions: {np.mean(percent_hals)*100}%, min: {np.min(percent_hals)*100}%, max: {np.max(percent_hals)*100}%, std: {np.std(percent_hals)*100}')
        logger.info(f'--------EVAL FINISHED--------')


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
    # tokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)
    tokenizer = PegasusTokenizerFast.from_pretrained(args.model_name, cache_dir=args.cache_path)
    basemodel = PegasusForConditionalGeneration.from_pretrained(args.model_name, max_position_embeddings = args.max_position_embeddings, cache_dir=args.cache_path)
    args.allnumber_path = '../support_files/allnumber_pegasus.pkl'
    # basemodel = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
    model = ModelSummaryMix(args, basemodel, tokenizer, args.model)
    promptnumber = args.prompt_number
    promptembedding = getpromptembedding(model, tokenizer, promptnumber, thistaskname, args.allnumber_path)
    model.set_prompt_embedding(promptnumber, promptembedding)
    # model weights
    if args.use_pretrain_ckpt and not(args.model in ["T5Finetune", "PegasusFinetune"]):
        ckptsum = torch.load(args.pretrain_ckpt, map_location="cuda:0")
        dicsum = {}
        for x in ckptsum.keys():
            if not (x in ["module.promptnumberforsum", "module.promptembeddingforsum"]):
                dicsum[x[7:]] = ckptsum[x]
        model.load_state_dict(dicsum)

    seed = 0
    args.few_shot_save_dir = args.data_dir + args.dataset + "/{}/".format(args.few_shot)
    
    ## LOAD CKPT
    args.model_save_folder = f'saved_models/{args.dataset}/{args.few_shot}/'
    if args.model != 'T5MixPrompt':
        args.model_save_folder += f'{args.model}/'
    # args.model_save_path = args.model_save_folder + f'seed_{seed}/'
    # path = args.model_save_path + 'bestckpt'
    # if args.counterfactual_trained:
    #     path = f'{path}_counterfactual'
    args.model_save_path = args.model_save_folder + f'seed_{args.seed}/'
    path = args.model_save_path + args.ckpt_name
    logger.info(f"loading checkpoint from {path}")
    ckptsum = torch.load(path, map_location="cuda:0")
    model.promptnumber = ckptsum["promptnumber"]
    model.promptembedding = nn.parameter.Parameter(ckptsum["promptembedding"])

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
    # if big dataset, change the validation set
    valid_file_name = args.data_dir + args.dataset + '/full_test.txt'
    args.full_testset = True

    # check if we have already generated it
    if not os.path.isfile(valid_file_name):
        dataset_args = [args.dataset_name, args.dataset_version]
        subsample_2k_testset(dataset_args, valid_file_name, args.seed, args)
    valid_dataset = SummarizationDataset(valid_file_name, "valid", args.max_length, tokenizer, allgentasktokens, answertoken, args)

    scaler = None
    spacy_nlp = spacy.load("en_core_web_sm")
    eval(model, valid_dataset, scaler, logger, args, tokenizer, spacy_nlp)

if __name__ == "__main__":
    args = set_args()
    set_logger(args)
    logger.info(args)
    main(args)


