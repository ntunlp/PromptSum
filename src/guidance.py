import operator
import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm



def spacy_ents_stats(data, split_name, spacy_nlp, args):
    print("\nRunning stats en spacy entities...")
    text_words_n = []
    text_ents_n = []
    text_ents_n_unique = []
    summary_words_n = []
    summary_ents_n = []
    summary_ents_n_unique = []
    summary_ents_n_in_text = []
    n_in_text_ratio = []
    summary_ents_n_unique_in_text = []
    n_unique_in_text_ratio = []
    for idx in tqdm(range(min(args.ents_stats_max_len, len(data)))):
        # text
        text_data = data[idx][args.text_key]
        n_words = len(word_tokenize(text_data))
        text_words_n.append(n_words)
        ents = spacy_nlp(text_data).ents
        text_unique_ents = []
        for x in ents:
            ent = x.text
            if not(ent in text_unique_ents):
                text_unique_ents.append(ent)
        text_ents_n.append(len(ents))
        text_ents_n_unique.append(len(text_unique_ents))
        # summary
        summary_data = data[idx][args.summary_key]
        n_words = len(word_tokenize(summary_data))
        summary_words_n.append(n_words)
        ents = spacy_nlp(summary_data).ents
        summary_unique_ents = []
        summary_ents_in_text = []
        summary_unique_ents_in_text = []
        for x in ents:
            ent = x.text
            if not(ent in summary_unique_ents):
                summary_unique_ents.append(ent)
                if ent in text_unique_ents:
                    summary_unique_ents_in_text.append(ent)
            if ent in text_unique_ents:
                summary_ents_in_text.append(ent)
        summary_ents_n.append(len(ents))
        summary_ents_n_unique.append(len(summary_unique_ents))
        summary_ents_n_in_text.append(len(summary_ents_in_text))
        if len(ents) > 0:
            n_in_text_ratio.append(100 * len(summary_ents_in_text) / len(ents))
        summary_ents_n_unique_in_text.append(len(summary_unique_ents_in_text))
        if len(summary_unique_ents) > 0:
            n_unique_in_text_ratio.append(100 * len(summary_unique_ents_in_text) / len(summary_unique_ents))
    print("\n", "STATS: {}".format(split_name))
    print("# text/summary pairs: {}".format(len(text_ents_n)))
    print("*"*50)
    print("Avg # words / text: {:.3f}".format(np.mean(text_words_n)))
    print("Avg # entities / text: {:.3f}".format(np.mean(text_ents_n)))
    print("Avg # unique entities / text: {:.3f}".format(np.mean(text_ents_n_unique)))
    print("*"*50)
    print("Avg # words / summary: {:.3f}".format(np.mean(summary_words_n)))
    print("Avg # entities / summary: {:.3f}".format(np.mean(summary_ents_n)))
    print("Avg # unique entities / summary: {:.3f}".format(np.mean(summary_ents_n_unique)))
    print("Avg # entities in summary which are in text: {:.3f}, as %: {:.3f}".format(np.mean(summary_ents_n_in_text), np.mean(n_in_text_ratio)))
    print("Avg # unique entities in summary which are in text: {:.3f}, as %: {:.3f}".format(np.mean(summary_ents_n_unique_in_text), np.mean(n_unique_in_text_ratio)))


def spacy_build_ents_frequency(data, spacy_nlp, max_len):
    ents_freq = {}
    for idx in tqdm(range(min(max_len, len(data)))):
        text_data = data[idx]['article']
        ents = spacy_nlp(text_data).ents
        for x in ents:
            ent = x.text
            if not (ent in ents_freq.keys()):
                ents_freq[ent] = 0
            ents_freq[ent] += 1
    ents_freq = dict(sorted(ents_freq.items(), key=operator.itemgetter(1), reverse=True))
    print("There are {} unique entities".format(len(ents_freq)))
    idx = 0
    for x in list(ents_freq.keys())[:20]:
        print("Entity ranked {} is {}, frequency: {} / {}".format(idx, x, ents_freq[x], len(ents_freq.keys())))
        idx += 1

    return ents_freq


def build_salient_sents(source, target, scorer, args):
    sents = sent_tokenize(source)
    all_rouges = []
    for sent in sents:
        rouge_scores = scorer.score(sent, target)
        r1 = 100 * rouge_scores["rouge1"].fmeasure
        all_rouges.append(r1)
    all_rouges = np.array(all_rouges)
    sort_idx = np.argsort(all_rouges)[::-1]
    sort_idx = sort_idx[:args.n_top_sents]
    top_sents = [sents[i] for i in sort_idx]

    return top_sents
