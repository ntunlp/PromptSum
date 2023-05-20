import pickle 
import numpy as np
from scipy.stats import ttest_ind



datasets = ["samsum"]
volumes = ["100"]
n_seeds = 3
metrics = ["r1s", "r2s", "rls", "bs"]
models = [
    "pretrained_False_oracle_False",
    "pretrained_True_oracle_False_no_finetuned_sprompt",
    "pretrained_True_oracle_False_no_sprompt",
    "pretrained_True_oracle_False_no_finetuned_eprompt",
    "pretrained_True_oracle_False_no_entity_chain"
]

for dataset in datasets:
    for volume in volumes:
        print(f"\nDataset: {dataset}, volume: {volume}")
        if volume == "full":
            base_scores = pickle.load(open(f"scores/{dataset}/{volume}/prompt_sum_scores_{dataset}_pretrained_True_oracle_False.pkl", "rb"))
            for model in models:
                path = f"scores/{dataset}/{volume}/prompt_sum_scores_{dataset}_{model}.pkl"
                other_scores = pickle.load(open(path, "rb"))
                print(f"Comparison with model {model}:")
                for metric in metrics:
                    stat, p_value = ttest_ind(base_scores[metric], other_scores[metric])
                    base = np.mean(base_scores[metric])
                    new = np.mean(other_scores[metric])
                    print(f"Metric {metric}, PromptSum: {base:.4f}, new: {new:.4f}, p-value: {p_value:.8f}")
        else:
            all_base_scores = []
            for j in range(n_seeds):
                base_scores = pickle.load(open(f"scores/{dataset}/{volume}/prompt_sum_scores_{dataset}_seed_{j}_pretrained_True_oracle_False.pkl", "rb"))
                all_base_scores.append(base_scores)
            base_scores = {}
            for k in all_base_scores[0].keys():
                base_scores[k] = np.mean(np.concatenate([np.expand_dims(x[k], 0) for x in all_base_scores]), 0)
            for model in models:
                all_other_scores = []
                for j in range(n_seeds):
                    path = f"scores/{dataset}/{volume}/prompt_sum_scores_{dataset}_seed_{j}_{model}.pkl"
                    other_scores = pickle.load(open(path, "rb"))
                    all_other_scores.append(other_scores)
                other_scores = {}
                for k in all_other_scores[0].keys():
                    other_scores[k] = np.mean(np.concatenate([np.expand_dims(x[k], 0) for x in all_other_scores]), 0)
                print(f"Comparison with model {model}:")
                for metric in metrics:
                    stat, p_value = ttest_ind(base_scores[metric], other_scores[metric])
                    base = np.mean(base_scores[metric])
                    new = np.mean(other_scores[metric])
                    print(f"Metric {metric}, PromptSum: {base:.4f}, new: {new:.4f}, p-value: {p_value:.8f}")
    

