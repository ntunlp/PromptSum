import pickle
import numpy as np

pegasus = pickle.load(open("data/cnndm_pegasus_50.pkl", "rb"))
promptsum = pickle.load(open("data/cnndm_promptsum_50.pkl", "rb"))

all_reversed = []
with open("data/cnndm_50.csv", "w") as f:
    f.write('Source document, Summary A, Summary B, Informative?, Factually consistent?, Relevant?, Fluent?, Coherent?')
    for i in range(len(pegasus['src'])):
        x = np.random.randint(2)
        f.write('\n' + "\"" + str(pegasus['src'][i]).replace("\"", "") + "\"")
        if x == 0:
            f.write(',' + "\"" + str(pegasus['model'][i]).replace("\"", "") + "\"")
            f.write(',' + "\"" + str(promptsum['model'][i]).replace("\"", "") + "\"")
            all_reversed.append(0)
        else:
            f.write(',' + "\"" + str(promptsum['model'][i]).replace("\"", "") + "\"")
            f.write(',' + "\"" + str(pegasus['model'][i]).replace("\"", "") + "\"")
            all_reversed.append(1)
        f.write(',' + ',' + ',' + ',' + ',')
print(all_reversed)

with open("permutation_csv", "wb") as f:
    pickle.dump(all_reversed, f)

