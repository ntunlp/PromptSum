import os
import pickle
import argparse
import gc
import time
gc.enable()



parser = argparse.ArgumentParser(description="latentRE")

parser.add_argument("--data_dir", type = str, default = "/export/home/dataset/PromptSumm/")
parser.add_argument("--dataset_name", type=str, default="ccdv/cnn_dailymail", 
                    choices = ["ccdv/cnn_dailymail", "xsum", "reddit_tifu", "wikihow", "billsum", "samsum"]) 
parser.add_argument("--dataset", type = str, default = "cnndm",
                    choices = ["cnndm", "xsum", "billsum", "samsum"])
parser.add_argument("--size", type = int, default = 64)
parser.add_argument("--seeds", type = list, default = [0])

args = parser.parse_args()



def main(args):
    dataset_names = ["ccdv/cnn_dailymail", "xsum", "reddit_tifu", "wikihow", "billsum", "samsum"]
    text_keys = ["article", "document", "documents", "text", "text", "dialogue"]
    summary_keys = ["highlights", "summary", "tldr", "headline", "summary", "summary"]

    idx = dataset_names.index(args.dataset_name)
    args.text_key = text_keys[idx]
    args.summary_key = summary_keys[idx]

    print(args)

    for seed in args.seeds:
        train_path = args.data_dir + args.dataset + "/{}/{}_few_shot_train_seed_{}".format(args.size, args.size, seed)
        valid_path = args.data_dir + args.dataset + "/{}/{}_few_shot_valid_seed_{}".format(args.size, args.size, seed)
        print(train_path, valid_path)

        train_data = pickle.load(open(train_path, "rb"))
        all_train_texts, all_train_summaries = [], []
        for idx in range(len(train_data)):
            text = train_data[idx][args.text_key]
            summary = train_data[idx][args.summary_key]
            summary = " ".join(summary.split("\n"))
            all_train_texts.append(text)
            all_train_summaries.append(summary)
        print(len(all_train_texts))
        if not os.path.exists(args.data_dir + args.dataset + "/{}/seed_{}".format(args.size, seed)):
            os.mkdir(args.data_dir + args.dataset + "/{}/seed_{}".format(args.size, seed))
        new_train_path = args.data_dir + args.dataset + "/{}/seed_{}/train.txt".format(args.size, seed)
        print("writing to: {}".format(new_train_path))
        with open(new_train_path, "w") as f:
            for idx in range(len(all_train_texts)):
                to_write = all_train_texts[idx] + "\t" + all_train_summaries[idx]
                if idx > 0:
                    to_write = "\n" + to_write
                f.write(to_write)

        valid_data = pickle.load(open(valid_path, "rb"))
        all_valid_texts, all_valid_summaries = [], []
        for idx in range(len(valid_data)):
            text = valid_data[idx][args.text_key]
            summary = valid_data[idx][args.summary_key]
            summary = " ".join(summary.split("\n"))
            all_valid_texts.append(text)
            all_valid_summaries.append(summary)
        print(len(all_valid_texts))
        new_valid_path = args.data_dir + args.dataset + "/{}/seed_{}/valid.txt".format(args.size, seed)
        print("writing to: {}".format(new_valid_path))
        with open(new_valid_path, "w") as f:
            for idx in range(len(all_valid_texts)):
                to_write = all_valid_texts[idx] + "\t" + all_valid_summaries[idx]
                if idx > 0:
                    to_write = "\n" + to_write
                f.write(to_write)


if __name__ == "__main__":
    main(args)







