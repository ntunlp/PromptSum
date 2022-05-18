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
    data = load_dataset('/export/home/PromptSumm/c4.py', 'realnewslike', cache_dir='/export/home/cache')
    data_fn = lambda x: find_salient_sentences_and_entities_per_example(x, scorer, spacy_nlp)
    new_data = data.map(data_fn, num_proc=multiprocessing.cpu_count()-4)
    new_data.save_to_disk("t5_tagger_pretraining_data/c4_realnewslike")

def convert_data():
    data = load_from_disk('t5_tagger_pretraining_data/c4_realnewslike')