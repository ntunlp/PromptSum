import pickle
import json
import csv

reversed = pickle.load(open("permutation_csv", "rb"))
print(reversed)
print(len(reversed))


with(open("cnndm_50_florian.csv", "r")) as f:
    reader = csv.reader(f)
    count = 0
    for l in reader:
        if count == 0:
            count += 1
            continue
        info = l[3]
        fact = l[4]
        rel = l[5]
        fluent = l[6]
        coh = l[7]
        d = {}
        d["informative"] = "a"
        if info == "A":
            d["informative"] = '1'
        if info == "B":
            d["informative"] = '2'
        d["factually consistent"] = "a"
        if fact == "A":
            d["factually consistent"] = '1'
        if fact == "B":
            d["factually consistent"] = '2'
        d["relevant"] = "a"
        if rel == "A":
            d["relevant"] = '1'
        if rel == "B":
            d["relevant"] = '2'
        d["fluent"] = "a"
        if fluent == "A":
            d["fluent"] = '1'
        if fluent == "B":
            d["fluent"] = '2'
        d["coherent"] = "a"
        if coh == "A":
            d["coherent"] = '1'
        if coh == "B":
            d["coherent"] = '2'

        print(d)
        if reversed[count-1] == 1:
            d['reversed_'] = 'True'
        else:
            d['reversed_'] = 'False'
        
        count += 1
        json.dump(d, open("../users/florian/result/{}.json".format(count-2), "w"))


