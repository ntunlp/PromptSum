import json
import numpy as np


dataset = "cnndm"
#dataset = "xsum"
users = ["ravox", "ruochen"]
#users = ["ravox", "ruochen"]
n = 50

all_counts = []
for user in users:
    print("*"*30, "User: {}".format(user))
    counts = []
    for i in range(n):
        path = "users/controllable_{}/{}/{}.json".format(dataset, user, i)
        res = json.load(open(path,"r"))
        satisfaction = res["satisfaction"]
        idx = 0
        while idx < len(satisfaction) and satisfaction[idx] != True:
            idx += 1
        counts.append(idx+1)
    counts = np.array(counts)
    for j in range(1, 5):
        count_j = np.sum(counts == j)
        print("# times user selects {}: {}".format(j, count_j))
