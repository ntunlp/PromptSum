import os
import shutil
directory = 'users'
outdirs = [dI for dI in os.listdir(directory) if os.path.isdir(os.path.join(directory,dI))]
for dir in outdirs:
    if os.path.isdir(directory + '/' + dir + '/result/'):
        shutil.rmtree(directory + '/' + dir + '/result/')
    os.mkdir(directory + '/' + dir + '/result/')
    if os.path.isdir(directory + '/' + dir + '/on_hold.json'):
        os.remove(directory + '/' + dir + '/on_hold.json')

import pickle
user_index = 0
user_index_dict = {'user_index': user_index}
with open("user_index.pkl", "w") as pkl_f:
    pickle.dump(user_index_dict, pkl_f)
