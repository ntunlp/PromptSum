import pickle
import argparse
import gc
gc.enable()

import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentRE")

    parser.add_argument("--data_dir", type = str, default = "/data/mathieu/DATASETS/PromptSumm/")
    parser.add_argument("--dataset", type = str, default = "cnndm")
    parser.add_argument("--size", type = int, default = 64)
    parser.add_argument("--seeds", type = list, default = [0])

    args = parser.parse_args()

    print(args)

    for seed in args.seeds:
        train_path = args.data_dir + args.dataset + "/{}/{}_few_shot_train_seed_{}".format(args.size, args.size, seed)
        valid_path = args.data_dir + args.dataset + "/{}/{}_few_shot_valid_seed_{}".format(args.size, args.size, seed)
        print(train_path, valid_path)

        train_data = pickle.load(open(train_path, "rb"))
        valid_data = pickle.load(open(valid_path, "rb"))
        print(type(train_data))








