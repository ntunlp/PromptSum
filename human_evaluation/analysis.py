import os
import json
import numpy as np



users = ["ravoxsamsum_0"]
n_summaries = 50

all_choices = [[], [], []]
for user in users:
    user_dir = "users/{}/result/".format(user)
    user_preferences, user_info, user_fluent, user_factual = [], [], [], []
    user_labels = []
    for i in range(n_summaries):
        summary_path = user_dir + "{}.json".format(i)
        with open(summary_path, "rb") as f:
            res_i = json.load(f)
            preference = res_i["preference"]
            if preference == "a":
                preference = 0
            else:
                preference = int(preference)
            user_preferences.append(preference)
            reason = res_i["reason"]
            info = 0
            fluent = 0
            factual = 0
            if reason in ["1", "4", "5", "7"]:
                info = 1
            if reason in ["2", "4", "6", "7"]:
                fluent = 1
            if reason in ["3", "5", "6", "7"]:
                factual = 1
            user_info.append(info)
            user_fluent.append(fluent)
            user_factual.append(factual)
            label = res_i["reversed_"]
            label = int(label == 'True')
            user_labels.append(label)
    user_preferences = np.array(user_preferences)
    user_info = np.array(user_info)
    user_fluent = np.array(user_fluent)
    user_factual = np.array(user_factual)
    user_labels = np.array(user_labels)
    user_new_model = []
    for i in range(len(user_labels)):
        if user_labels[i] == 0:
            user_new_model.append(2)
        else:
            user_new_model.append(1)
    user_new_model = np.array(user_new_model)
    print("\nUser: {}".format(user))
    for j in [0,1,2]:
        print(j, np.sum(user_preferences == j))
    user_choice = []
    for i in range(len(user_new_model)):
        if user_preferences[i] == 0:
            user_choice.append(0)
        else:
            if user_new_model[i] != user_preferences[i]:
                user_choice.append(1)
            else:
                user_choice.append(2)
    user_choice = np.array(user_choice)
    base_idx = np.arange(len(user_choice))[user_choice == 1]
    new_idx = np.arange(len(user_choice))[user_choice == 2]

    tie_perc = 100 * np.sum(user_preferences == 0) / n_summaries
    all_choices[0].append(tie_perc)
    print("% User chooses tie: {} ({:.4f}%)".format(np.sum(user_preferences==0), tie_perc))

    old_model = np.sum(user_new_model != user_preferences) - np.sum(user_preferences == 0)
    old_model_perc = 100 * old_model / n_summaries
    all_choices[1].append(old_model_perc)
    print("% User chooses baseline: {} ({:.4f}%)".format(old_model, old_model_perc))
    base_info = np.sum(user_info[base_idx] == 1)
    base_fluent = np.sum(user_fluent[base_idx] == 1)
    base_factual = np.sum(user_factual[base_idx] == 1)
    print("Including # informative: {}, # fluent: {}, # factually correct: {}".format(base_info, base_fluent, base_factual))

    new_model = np.sum(user_new_model == user_preferences)
    new_model_perc = 100 * new_model / n_summaries
    all_choices[2].append(new_model_perc)
    print("% User prefers new model: {} ({:.4f}%)".format(new_model, new_model_perc))
    new_info = np.sum(user_info[new_idx] == 1)
    new_fluent = np.sum(user_fluent[new_idx] == 1)
    new_factual = np.sum(user_factual[new_idx] == 1)
    print("Including # informative: {}, # fluent: {}, # factually correct: {}".format(new_info, new_fluent, new_factual))

for j in range(len(all_choices)):
    all_choices[j] = np.array(all_choices[j])
m_tie = np.mean(all_choices[0])
std_tie = np.std(np.array(all_choices[0]))
print("\n")
print("Mean fraction of tie: {:.4f}%, std: {:.4f}".format(m_tie, std_tie))
m_old = np.mean(all_choices[1])
std_old = np.std(all_choices[1])
print("Mean fraction of old model (PEGASUS) preference: {:.4f}%, std: {:.4f}".format(m_old, std_old))
m_new = np.mean(all_choices[2])
std_new = np.std(all_choices[2])
print("Mean fraction of new model (SummaFusion) preference: {:.4f}%, std: {:.4f}".format(m_new, std_new))
