from rouge_score import rouge_scorer
import nltk
from nltk.tokenize import sent_tokenize
import spacy
from engine_pretrain import find_salient_sentences_and_entities_per_example
from datasets import load_dataset, load_from_disk
import multiprocessing

def process_c4():
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    spacy_nlp = spacy.load("en_core_web_sm")
    data = load_dataset('/export/home/PromptSumm/c4.py', 'realnewslike_v2', cache_dir='/export/home/cache')
    data_fn = lambda x: find_salient_sentences_and_entities_per_example(x, scorer, spacy_nlp)
    new_data = data.map(data_fn, num_proc=multiprocessing.cpu_count()-4)
    new_data.save_to_disk("t5_tagger_pretraining_data/realnewslike_v2")

def filter_bad_c4():
    data = load_from_disk("t5_tagger_pretraining_data/c4_realnewslike")
    print(f'raw data rows: {len(data["train"])}')
    data['train'] = data['train'].filter(lambda x: len(x['ent_chain']) > 1, num_proc=multiprocessing.cpu_count()-4)
    print(f'clean data rows: {len(data["train"])}')
    # import pdb;pdb.set_trace()
    # data.save_to_disk("t5_tagger_pretraining_data/c4_realnewslike_clean")

def subsample_validation():
    data = load_from_disk("t5_tagger_pretraining_data/c4_realnewslike")
    test_valid = data['validation'].train_test_split(test_size=1000)
    data['val_small'] = test_valid['test']
    data['val_small'].save_to_disk("t5_tagger_pretraining_data/c4_realnewslike_v2/val_small")

if __name__ == '__main__':
    # process_c4()
    filter_bad_c4()
    # subsample_validation()
    # 2695280 -> 2695280(summary)  -> 2571483(text_rest>10) -> 2695280(ent_chain)