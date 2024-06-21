# This script performs qualitative tests to test a model for controllablity: success rates
import argparse

from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, PegasusTokenizerFast
from fairseq.models.bart import BARTModel

from hyperparameters import root, cache_path, pretrain_ckpt, pretrain_prompt_ckpt
from utils import settle_dataset_args
from dataset.dataset_entity import DatasetEntity
from dataset.dataset_summary import DatasetSummary
from engine_pretrain import *
from engine_entity import *
from engine_summary import *
from models.model_summary_finetune import *
from models.model_summary_soft import *
from models.model_summary_mix import *


def set_args():
    parser = argparse.ArgumentParser(description="latentRE")

    parser.add_argument("--data_dir", dest="data_dir", type=str,
                        default= root + "DATASETS/PromptSum/")
    parser.add_argument("--CTRLsum_ckpt_dir", dest="CTRLsum_ckpt_dir", type=str,
                        default='/data/mathieu/prompt_sum/src/saved_models/xsum/100/CTRLsum_origin/')
    parser.add_argument("--cuda", dest="cuda", type=str,
                        default="2", help="gpu id") 
    parser.add_argument("--tune_weights", dest="tune_weights", action='store_true',
                        default=False)
    parser.add_argument("--ckpt_name", dest="ckpt_name", type=str,
                        default="full_weights", help="model ckpt name")                     
    parser.add_argument("--dataset_name", dest="dataset_name", type=str,
                        default="xsum")
    parser.add_argument("--model", dest="model", type=str,
                        default="CTRLsum_origin", choices = ["PegasusFinetune", 'PegasusSoftPrompt', 'PegasusMixPrompt', 'CTRLsum', "CTRLsum_origin"])
    parser.add_argument("--model_name", dest="model_name", type=str,
                        default="facebook/bart-large", choices=["t5-base", "google/t5-v1_1-base", "facebook/bart-base",
                        "facebook/bart-large", "google/pegasus-large"])
    parser.add_argument("--use_lm_adapted", dest="use_lm_adapted", type=int,
                        default=0, help="whether to use lm_adapted model") #if we use bart, then automatically don't use lm_adapted
    parser.add_argument("--lm_adapted_path", dest="lm_adapted_path", type=str,
                        default=root + "lm_adapted_t5model/torch_ckpt/large/pytorch_model.bin",
                        help="The path of lm_adapted model")
    parser.add_argument("--cache_path", dest="cache_path", type=str,
                        default=cache_path, help="The path of huggingface cache")
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
                        default='100', help="number of data points for training AND validation")
    parser.add_argument("--use_t5_tagger",  action='store_false',
                        default=True, help="whether use a t5 tagger")
    parser.add_argument("--infer_val_entities", action="store_false",
                        default=True, help="whether to run inference with the T5 entity chain prediction on val set")
    parser.add_argument("--pretrain", action='store_true',
                        default=False, help="whether pretrain a T5 tagger")
    parser.add_argument("--num_beams", dest="num_beams", type=int,
                        default=4, help="number of beams in beam search")
    parser.add_argument("--repetition_penalty", dest="repetition_penalty", type=float,
                        default=1.0, help="repetition penalty")
    parser.add_argument("--length_penalty", dest="length_penalty", type=float,
                        default=1.0, help="length penalty")
    parser.add_argument("--stemmer", dest="stemmer", type=bool, 
                        default=True)
    parser.add_argument("--prompt_number", dest="prompt_number", type=int,
                        default=300, help="The number of prompt")
    parser.add_argument("--use_pretrain_ckpt", action='store_false',
                        default=True, help="whether to load the pre-training ckpt before fine-tuning")
    parser.add_argument("--pretrain_ckpt", type=str,
                        default=pretrain_ckpt, help="path to pretrained model")
    parser.add_argument("--pretrain_prompt_ckpt", type=str,
                        default=pretrain_prompt_ckpt, help="path to pretrained model prompt")
    parser.add_argument("--full_testset", action='store_true', help="whether or not to evaluate using the full testset")
    parser.add_argument("--seed", dest="seed", type=int,
                        default=0, help="seed for network")
    
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    parser.add_argument("--max_length_entity", type=int, default=128)
    parser.add_argument("--diversity_dbs", type=bool, default=True)
    parser.add_argument("--diversity_entity", type=bool, default=False)

    parser.add_argument('--num_beam_groups', type=int, default=10)  # default: 10
    parser.add_argument('--diversity_penalty', type=float, default=1.0)  # default: 1.0
    parser.add_argument('--num_diverse_beams', type=int, default=10)
    parser.add_argument('--num_return_sequences', type=int, default=10)
    
    args = parser.parse_args()
    settle_dataset_args(args)

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
        args.few_shot_save_dir = args.data_dir + args.dataset + f"/{args.few_shot}/"
    else:
        basemodel = PegasusForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
        tokenizer = PegasusTokenizerFast.from_pretrained(args.model_name, cache_dir=args.cache_path)
        args.allnumber_path = 'support_files/allnumber_pegasus.pkl'
        if "Finetune" in args.model:
            model = ModelSummaryFinetune(args, basemodel, tokenizer, args.model)
        elif "Mix" in args.model:
            model = ModelSummaryMix(args, basemodel, tokenizer, args.model)
            promptnumber = args.prompt_number
            promptembedding = getpromptembedding(model, tokenizer, promptnumber, thistaskname, args.allnumber_path)
            model.set_prompt_embedding(promptnumber, promptembedding)
        # model weights
        if args.use_pretrain_ckpt and not(args.model in ['T5Finetune', 'BartFinetune', 'PegasusFinetune']):
            print(f"device: {args.device}")
            ckptsum = torch.load(args.pretrain_ckpt, map_location=args.device)
            dicsum = {}
            for x in ckptsum.keys():
                if not (x in ["module.promptnumberforsum", "module.promptembeddingforsum"]):
                    dicsum[x[7:]] = ckptsum[x]
            model.load_state_dict(dicsum)
        logger.info('loaded model')

        args.few_shot_save_dir = args.data_dir + args.dataset + f"/{args.few_shot}/"
        
        ## LOAD CKPT
        args.model_save_folder = f'saved_models/{args.dataset}/{args.few_shot}/'
        args.model_save_folder += f'{args.model}/'
        args.model_save_path = args.model_save_folder + f'seed_{args.seed}/'
        path = args.model_save_path + args.ckpt_name
        ckptsum = torch.load(path)
        print(f"Loading from : {path}")
        if 'full_weights' in args.ckpt_name or (args.model in ['T5Finetune', 'BartFinetune', 'PegasusFinetune']):
            model.load_state_dict(ckptsum)
            print("HERE")
        else:
            model.promptnumber = ckptsum["promptnumber"]
            model.promptembedding = nn.parameter.Parameter(ckptsum["promptembedding"])

        logger.info('loaded ckpt')
        few_shot_seeds = [0]
        for gg in range(len(allgentasktokens)):
            gentasktoken = allgentasktokens[gg]
            tokenizer.add_tokens(gentasktoken)
            logger.info(f'gen token = {gentasktoken} , gen token id = {tokenizer.convert_tokens_to_ids(gentasktoken)}')
        answertoken = "__ans__"
        special_tokens = {"ans_token": answertoken}
        tokenizer.add_tokens(list(special_tokens.values()))
        tokenizer.add_tokens(['[SEP]'])
    model = model.to(args.device)

    if "Mix" in args.model:
        entbasemodel = PegasusForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_path)
        enttokenizer = PegasusTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_path)
        entmodel = ModelEntity(entbasemodel, enttokenizer, args)
        if args.use_pretrain_ckpt:
            ckpt = torch.load(args.pretrain_ckpt, map_location="cuda:0")
            dic = {}
            for x in ckpt.keys():
                if (args.max_position_embeddings > 1024) and ("embed_positions" in x):
                    continue
                if not (x in ["module.promptnumber", "module.promptembedding", "module.promptnumberforsum",
                              "module.promptembeddingforsum"]):
                    dic[x[7:]] = ckpt[x]
            if args.max_position_embeddings > 1024:
                dic["model.model.encoder.embed_positions.weight"] = entbasemodel.state_dict()[
                    "model.encoder.embed_positions.weight"]
                dic["model.model.decoder.embed_positions.weight"] = entbasemodel.state_dict()[
                    "model.decoder.embed_positions.weight"]
            entmodel.load_state_dict(dic)
            logger.info("Loaded the pre-trained ckpt for the entity prediction model!")

            onepath = f'entity_ckpt/{args.dataset}/{args.few_shot}/seed_{args.seed}/bestckpt_prompt'
            if args.use_pretrain_ckpt:
                onepath += "_from_pretrained"
            oneckpt = torch.load(onepath)
            entmodel.promptnumber = oneckpt["promptnumber"]
            entmodel.promptembedding = oneckpt["promptembedding"]
        entmodel = entmodel.to(args.device)

    # use the whole testset
    valid_file_name = args.data_dir + args.dataset + '/full_test.txt'
    print(valid_file_name)
    args.logger = logger
    logger.info('generated dataset')
    valid_dataset = DatasetSummary(valid_file_name, "valid", args.max_length, tokenizer, allgentasktokens, answertoken, args, human_eval = True)
    print("\n1st data point:")
    print(valid_dataset.data[0][0][:500])

    # generation
    n_gen = 5
    n_chains = 10
    n_entities = 3
    n_beams = 10

    all_summaries, all_labels = [], []
    print(len(valid_dataset.data))

    # collect all entities first (for CTRLSum)
    full_ents = []
    for i, (inp, tar) in tqdm(enumerate(valid_dataset.data[:100])):
        inp = ' '.join(inp.split()[:400])  # mimic truncation
        sents = sent_tokenize(inp)
        if len(sents) > 3:
            full = ''.join(sents)
        else:
            full = ''.join(sents)
        fullents = valid_dataset.spacy_nlp(full).ents
        fullents = [ent.text for ent in fullents]
        full_ents.append(fullents)
    
    # 1 - just DBS
    if args.diversity_dbs and not(args.diversity_entity):
        print("\nDiversity based on DBS")
        for i in tqdm(range(n_gen)):
            inp, tar = valid_dataset.data[i]

            # target
            all_labels.append(tar)
            target_res = valid_dataset.tokenizer.batch_encode_plus([tar], padding=False, max_length=valid_dataset.maxlen, truncation=True, return_tensors="pt")
            target_ids = target_res['input_ids']
            target_attn_mask = target_res['attention_mask']

            # source
            input_res = valid_dataset.tokenizer.batch_encode_plus([inp], padding=False, max_length=valid_dataset.maxlen, truncation=True, return_tensors="pt")
            input_ids = input_res['input_ids']
            input_attn_mask = input_res['attention_mask']

            if "CTRLsum" in args.model:
                import random
                original_ents = full_ents[i]
                original_ents = list(set([ent for ent in original_ents if '.' not in ent]))
                sample_iter, sampled_ent_tuples = 0, set()
                while len(sampled_ent_tuples) < 20 and sample_iter < 60:
                    ents = tuple(random.sample(original_ents, 3))
                    if ents not in sampled_ent_tuples:
                        sampled_ent_tuples.add(ents)
                    sample_iter += 1

                new_data = []
                original_data = valid_dataset.data[i]
                for cur_ents in sampled_ent_tuples:
                    ent_str = ' | '.join(cur_ents)
                    new_text = f'{ent_str} => {original_data[0]}'
                    new_data.append([new_text, original_data[1]])
                    break

                _, preds = eval_ctrlsum(model, tokenizer, new_data, logger, dbs=True)
                print(preds)
                raise Exception

            else:
                if "Mix" in args.model:
                    # prediction with entities model
                    inputs = {"input_ids": input_ids.to(args.device), "attention_mask": input_attn_mask.to(args.device)}
                    _, _, tagpreds = entmodel._generative_step(inputs)
                    ent_chain = tagpreds[0]
                else:
                    ent_chain = "None"

                # encode entities
                ent_res = valid_dataset.tokenizer.batch_encode_plus([ent_chain], padding=False, max_length=valid_dataset.maxlen, truncation=True, return_tensors="pt")
                ent_ids = ent_res['input_ids']
                ent_attn_mask = ent_res['attention_mask']

                # summary prediction
                inputs = {"input_ids": input_ids.to(args.device), "attention_mask": input_attn_mask.to(args.device),
                        "target_ids": target_ids.to(args.device), "target_mask": target_attn_mask.to(args.device),
                        "ents_ids": ent_ids.to(args.device), "ents_mask": ent_attn_mask.to(args.device)}
                sen, target, preds = model._diverse_generative_step(inputs)
                all_summaries.append(preds)

    # 2 - just entities
    if not(args.diversity_dbs) and args.diversity_entity:
        print("\nDiversity based on entities")
        for i in tqdm(range(n_gen)):
            inp, tar = valid_dataset.data[i]

            # target
            all_labels.append(tar)
            target_res = valid_dataset.tokenizer.batch_encode_plus([tar], padding=False, max_length=valid_dataset.maxlen, truncation=True, return_tensors="pt")
            target_ids = target_res['input_ids']
            target_attn_mask = target_res['attention_mask']

            # source
            input_res = valid_dataset.tokenizer.batch_encode_plus([inp], padding=False, max_length=valid_dataset.maxlen, truncation=True, return_tensors="pt")
            input_ids = input_res['input_ids']
            input_attn_mask = input_res['attention_mask']

            # entities
            _inp = ' '.join(inp.split())
            _inp_ents = valid_dataset.spacy_nlp(_inp).ents
            _inp_ents_text = [ent.text for ent in _inp_ents]
            new_ents_text = []
            for x in _inp_ents_text:
                if not(x in new_ents_text):
                    new_ents_text.append(x)
            _inp_ents_text = new_ents_text
            if len(_inp_ents_text) == 0:
                _inp_ents_text = ["None"]

            summaries = []
            for j in range(n_chains):
                ent_chain = []
                for k in range(n_entities):
                    ent = _inp_ents_text[np.random.randint(len(_inp_ents_text))]
                    ent_chain.append(ent)
                ent_chain = ",".join(ent_chain)
                ent_res = valid_dataset.tokenizer.batch_encode_plus([ent_chain], padding=False, max_length=valid_dataset.maxlen, truncation=True, return_tensors="pt")
                ent_ids = ent_res['input_ids']
                ent_attn_mask = ent_res['attention_mask']

                inputs = {"input_ids": input_ids.to(args.device), "attention_mask": input_attn_mask.to(args.device),
                          "target_ids": target_ids.to(args.device), "target_mask": target_attn_mask.to(args.device),
                          "ents_ids": ent_ids.to(args.device), "ents_mask": ent_attn_mask.to(args.device)}
                sen, target, preds = model._generative_step(inputs)
                summaries.append(preds[0])
            all_summaries.append(summaries)

    # 3 - DBS + entities
    if args.diversity_dbs and args.diversity_entity:
        print("\nDiversity based on DBS + entities")
        for i in tqdm(range(n_gen)):
            inp, tar = valid_dataset.data[i]

            # target
            all_labels.append(tar)
            target_res = valid_dataset.tokenizer.batch_encode_plus([tar], padding=False, max_length=valid_dataset.maxlen, truncation=True, return_tensors="pt")
            target_ids = target_res['input_ids']
            target_attn_mask = target_res['attention_mask']

            # source
            input_res = valid_dataset.tokenizer.batch_encode_plus([inp], padding=False, max_length=valid_dataset.maxlen, truncation=True, return_tensors="pt")
            input_ids = input_res['input_ids']
            input_attn_mask = input_res['attention_mask']

            # entities
            _inp = ' '.join(inp.split())
            _inp_ents = valid_dataset.spacy_nlp(_inp).ents
            _inp_ents_text = [ent.text for ent in _inp_ents]
            new_ents_text = []
            for x in _inp_ents_text:
                if not(x in new_ents_text):
                    new_ents_text.append(x)
            _inp_ents_text = new_ents_text
            if len(_inp_ents_text) == 0:
                _inp_ents_text = ["None"]

            summaries = []
            for j in range(n_chains):
                ent_chain = []
                for k in range(n_entities):
                    ent = _inp_ents_text[np.random.randint(len(_inp_ents_text))]
                    ent_chain.append(ent)
                ent_chain = ",".join(ent_chain)
                ent_res = valid_dataset.tokenizer.batch_encode_plus([ent_chain], padding=False, max_length=valid_dataset.maxlen, truncation=True, return_tensors="pt")
                ent_ids = ent_res['input_ids']
                ent_attn_mask = ent_res['attention_mask']

                inputs = {"input_ids": input_ids.to(args.device), "attention_mask": input_attn_mask.to(args.device),
                          "target_ids": target_ids.to(args.device), "target_mask": target_attn_mask.to(args.device),
                          "ents_ids": ent_ids.to(args.device), "ents_mask": ent_attn_mask.to(args.device)}
                sen, target, preds = model._diverse_generative_step(inputs)
                summaries += preds
            all_summaries.append(summaries)

    print(f"\nExtracted {len(all_summaries[0])} summaries / data point")

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    # random and oracle R-1
    all_random, all_oracles = [], []
    for i in tqdm(range(len(all_summaries))):
        summaries = all_summaries[i]
        label = all_labels[i]
        label = "\n".join(sent_tokenize(label))
        r1s = []
        for j in range(len(summaries)):
            summary = summaries[j]
            summary = "\n".join(sent_tokenize(summary))
            rouge_scores = scorer.score(label, summary)
            r1 = 100 * rouge_scores["rouge1"].fmeasure
            r1s.append(r1)
        r1s = np.array(r1s)
        random = r1s[np.random.randint(len(r1s))]
        all_random.append(random)
        oracle = np.max(r1s)
        all_oracles.append(oracle)
    all_random = np.array(all_random)
    print(f"Random score: {np.mean(all_random):.4f}, std: {np.std(all_random):.4f}")
    all_oracles = np.array(all_oracles)
    print(f"Oracle score: {np.mean(all_oracles):.4f}, std: {np.std(all_oracles):.4f}")

    # inter-ROUGE
    all_inter = []
    for i in tqdm(range(len(all_summaries))):
        summaries = all_summaries[i]
        r1s = []
        for j in range(len(summaries)):
            summary_j = summaries[j]
            summary_j = "\n".join(sent_tokenize(summary_j))
            for k in range(j+1, len(summaries)):
                summary_k = summaries[k]
                summary_k = "\n".join(sent_tokenize(summary_k))
                rouge_scores = scorer.score(summary_j, summary_k)
                r1_j_k = 100 * rouge_scores["rouge1"].fmeasure
                rouge_scores = scorer.score(summary_k, summary_j)
                r1_k_j = 100 * rouge_scores["rouge1"].fmeasure
                r1 = (r1_j_k + r1_k_j) / 2
                r1s.append(r1)
        r1s = np.array(r1s)
        r1 = np.mean(r1s)
        all_inter.append(r1)
    all_inter = np.array(all_inter)
    print(f"Inter-candidates score: {np.mean(all_inter):.4f}, std: {np.std(all_inter):.4f}")

def eval_ctrlsum(model, tokenizer, data, logger, dbs=False):
    '''
    Args:
        data: list of source texts
        ents: list of control entities, one item for each source text
    '''
    model.cuda()
    model.eval()
    # model.half()
    batch_size = args.valid_size_per_gpu_summary

    all_inputs = []
    allypred = []
    with torch.no_grad():
        logger.info(f"inf {len(data)} data points")
        for i in range(0, len(data), batch_size):  # function as batch iterator
            batch_inputs = data[i:i + batch_size]
            batch_inputs = [v[0] for v in batch_inputs]

            if dbs:
                res = model.sample(batch_inputs, nbest=4)
                print(res)
                raise Exception
            else:
                res = model.sample(batch_inputs, beam=4, prefix_tokens=None, lenpen=1.0, max_len_b=140, min_len=1, no_repeat_ngram_size=3, extra_gen_cls_kwargs=None)
            allypred.extend(res)
            all_inputs.extend(batch_inputs)
            if i % 20 == 0:
                logger.info(f"inf, {i} data points processed")

    return all_inputs, allypred


if __name__ == "__main__":
    args = set_args()
    set_logger(args)
    logger.info(args)
    seed_everything(args)
    main(args)


