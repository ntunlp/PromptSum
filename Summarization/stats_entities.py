import argparse
import spacy
from tqdm import tqdm

from utils import *
from dataset.dataset import *



parser = argparse.ArgumentParser(description="latentRE")

##### data
# For the following argument, follow the order "cnndm", "xsum", "reddit", "wikihow", "billsum", "samsum"
parser.add_argument("--dataset_name", dest="dataset_name", type=str,
                    default="ccdv/cnn_dailymail", help="data name",
                    choices = ["ccdv/cnn_dailymail", "xsum", "reddit_tifu", "wikihow", "billsum", "samsum"])
parser.add_argument("--ents_stats_max_len", type=int, default=100000)

args = parser.parse_args()

dataset_names = ["ccdv/cnn_dailymail", "xsum", "reddit_tifu", "wikihow", "billsum", "samsum"]

idx = dataset_names.index(args.dataset_name)
if args.dataset_name == 'cnn_dailymail' or args.dataset_name == "ccdv/cnn_dailymail":
    idx = 0
    args.dataset = 'cnndm'
else:
    args.dataset = args.dataset_name

# print args
print(args)



def main(args):
    # set seed
    seed_everything(args)

    # data
    spacy_nlp = spacy.load("en_core_web_sm")
    sets = ["test"]
    for set in sets:
        path = "../../DATASETS/PromptSumm/{}/full/seed_42/{}.txt".format(args.dataset, set)
        print(path)
        count = 0
        source_ents, summary_ents = [], []
        with open(path, 'r') as f:
            for l in tqdm(f.readlines()):
                data = l.split("\t")
                source = data[0]
                ents = spacy_nlp(source).ents
                allents = [ent.text for ent in ents]
                source_ents.append(len(allents))
                summary = data[1]
                ents = spacy_nlp(summary).ents
                allents = [ent.text for ent in ents]
                summary_ents.append(len(allents))

                count += 1
                if count >= args.ents_stats_max_len:
                    break
        source_ents = np.array(source_ents)
        summary_ents = np.array(summary_ents)
        print("Average # entities in the source: {:.4f}, std: {:.4f}".format(np.mean(source_ents), np.std(source_ents)))
        print("Average # entities in the summary: {:.4f}, std: {:.4f}".format(np.mean(summary_ents), np.std(summary_ents)))



if __name__ == '__main__':
    main(args)

