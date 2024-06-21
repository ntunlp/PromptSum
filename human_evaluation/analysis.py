import os
import json
import numpy as np


dataset = "generic_cnndm"
users = ["chengwei", "hailin", "ravox", "ruochen"]
n_summaries = 50

attributes = ["informative", "factually consistent", "relevant", "fluent", "coherent"]
all_user_info, all_user_fact, all_user_rel, all_user_fluent, all_user_coh = [], [], [], [], []
for user in users:
    user_dir = "users/{}/{}/result/".format(dataset, user)
    user_info, user_fact, user_rel, user_fluent, user_coh = [], [], [], [], []
    user_labels = []
    print("*"*50 + " User: {}".format(user))
    for i in range(n_summaries):
        summary_path = user_dir + "{}.json".format(i)
        with open(summary_path, "rb") as f:
            res_i = json.load(f)
            # informative
            info = res_i["informative"]
            if info == "a":
                info = 0
            else:
                info = int(info)
            user_info.append(info)
            # factually consistent
            fact = res_i["factually consistent"]
            if fact == "a":
                fact = 0
            else:
                fact = int(fact)
            user_fact.append(fact)
            # relevant
            rel = res_i["relevant"]
            if rel == "a":
                rel = 0
            else:
                rel = int(rel)
            user_rel.append(rel)
            # fluent 
            fluent = res_i["fluent"]
            if fluent == "a":
                fluent = 0
            else:
                fluent = int(fluent)
            user_fluent.append(fluent)
            # coherent
            coh = res_i["coherent"]
            if coh == "a":
                coh = 0
            else:
                coh = int(coh)
            user_coh.append(coh)

            label = res_i["reversed_"]
            label = int(label == 'True')
            user_labels.append(label)
    user_info = np.array(user_info)
    user_fact = np.array(user_fact)
    user_rel = np.array(user_rel)
    user_fluent = np.array(user_fluent)
    user_coh = np.array(user_coh)
    user_labels = np.array(user_labels)
    all_user_info.append(user_info)
    all_user_fact.append(user_fact)
    all_user_rel.append(user_rel)
    all_user_fluent.append(user_fluent)
    all_user_coh.append(user_coh)

    not_reversed_idx = list(np.arange(len(user_labels))[user_labels == 0])
    reversed_idx = list(np.arange(len(user_labels))[user_labels == 1])
    print("# not reversed: {}, # reversed: {}".format(len(not_reversed_idx), len(reversed_idx)))
    arrs = [user_info, user_fact, user_rel, user_fluent, user_coh]
    for j in range(len(arrs)):
        arr = arrs[j]
        pegasus_count, promptsum_count, tie_count = 0, 0, 0
        for i in range(len(arr)):
            if arr[i] == 0:
                tie_count += 1
            elif arr[i] == 1:
                if i in not_reversed_idx:
                    pegasus_count += 1
                else:
                    promptsum_count += 1
            else:
                if i in not_reversed_idx:
                    promptsum_count += 1
                else:
                    pegasus_count += 1
        print("Aspect {}, PEGASUS count: {}, PromptSum: {}, Tie: {}".format(
            attributes[j], pegasus_count, promptsum_count, tie_count))
    
arrs = [all_user_info, all_user_fact, all_user_rel, all_user_fluent, all_user_coh]
for k in range(len(arrs)):
    arr = arrs[k]
    agree = 0
    for i in range(len(arr[0])):
        preds = []
        for j in range(len(arr)):
            preds.append(arr[j][i])
        preds = np.array(preds)
        els = np.unique(preds)
        counts = np.array([np.sum(preds == el) for el in els])
        max_count = np.max(counts) 
        ag = 100 * max_count / len(arr)
        agree += ag
    agree /= len(arr[0])
    print("Attribute {}, agreement between humans: {:.4f}".format(attributes[k], agree))


