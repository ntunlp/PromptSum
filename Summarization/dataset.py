import datasets
import os
import numpy as np



def subsample(dataset_args, few_shot_seeds, args):
    '''
    Function that subsamples a dataset and saves the results for few-shot exps
    args:
        few_shot_seeds: list of random seeds to produce the subsamples repeatively
    '''
    if args.dataset_name == "billsum":
        data = datasets.load_dataset(*dataset_args, download_mode="force_redownload", cache_dir=args.dataset_cache_dir)
        test_data = data['test']
        x_data = data['train'].train_test_split(test_size=0.1, shuffle=True)
        train_data = x_data['train']
        valid_data = x_data['test']
    else:
        data = datasets.load_dataset(*dataset_args, cache_dir=args.dataset_cache_dir)
        train_data = data['train']
        valid_data = data['validation']
        test_data = data['test']
    print("\nTotal size: {}".format(len(data)))
    len_train = len(train_data)
    len_valid = len(valid_data)
    for seed in few_shot_seeds:
        os.makedirs(args.few_shot_save_dir + 'seed_{}'.format(seed), exist_ok=True)
        # re-set random seed
        np.random.seed(seed)
        indices = np.random.choice(range(len_train), args.few_shot)
        train_data_new = train_data.select(indices)
        indices = np.random.choice(range(len_valid), args.few_shot)
        valid_data_new = valid_data.select(indices)
        # save
        train_path = args.few_shot_save_dir + 'seed_{}/train.txt'.format(seed)
        valid_path = args.few_shot_save_dir + 'seed_{}/valid.txt'.format(seed)
        test_path = args.few_shot_save_dir + 'seed_{}/test.txt'.format(seed)
        convert_data_to_txt(train_data_new, train_path, args)
        convert_data_to_txt(valid_data_new, valid_path, args)
        convert_data_to_txt(test_data, test_path, args)
    # convert to original seed
    np.random.seed(args.seed)


def subsample_2k_testset(dataset_args, file_path, seed, args, n = 2000, valid = False):
    '''
    Function that subsamples a 2k test set that can be reused
    args:
        file_path: directory to save the testset
        seed: random seed to sample with
    '''
    data = datasets.load_dataset(*dataset_args, cache_dir=args.dataset_cache_dir)
    if args.dataset_name == "billsum":
        x_data = data['train'].train_test_split(test_size=0.1, shuffle=True)
        valid_data = x_data['test']
    else:
        valid_data = data['validation']
    if not(valid):
        valid_data = data['test']

    if args.full_testset:
        convert_data_to_txt(valid_data, file_path, args)
    else:
        len_valid = len(valid_data)
        np.random.seed(seed)
        indices = np.random.choice(range(len_valid), n)
        valid_data_new = valid_data.select(indices)
        # save
        convert_data_to_txt(valid_data_new, file_path, args)
        # convert to original seed
        np.random.seed(args.seed)


def convert_data_to_txt(train_data, new_train_path, args):
    all_train_texts, all_train_summaries = [], []
    none_texts = 0
    for idx in range(len(train_data)):
        text = train_data[idx][args.text_key]
        text = " ".join(text.split("\n"))
        if text in ["", " ", "  "]:
            text = "None"
            none_texts += 1
        summary = train_data[idx][args.summary_key]
        summary = " ".join(summary.split("\n"))
        all_train_texts.append(text)
        all_train_summaries.append(summary)
    print("writing to: {}".format(new_train_path))
    print("Total size: {}".format(len(all_train_texts)))
    print("Missing sources: {}".format(none_texts))
    with open(new_train_path, "w") as f:
        for idx in range(len(all_train_texts)):
            if args.dataset_name == "samsum":
                to_write = all_train_texts[idx].replace("\n", " ").replace("\t", " ").replace("\r", " ") + "\t" + all_train_summaries[idx].replace("\n", " ").replace("\t", " ").replace("\r", " ")
            else:
                to_write = all_train_texts[idx] + "\t" + all_train_summaries[idx]
            if idx > 0:
                to_write = "\n" + to_write
            f.write(to_write)
