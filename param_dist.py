from utils import * 
import os 
from tqdm import tqdm
import glob

init = 0
lr = 0.1
bs = 64
neurons = 50
max_iter = 200000
ss = [i for i in range(0, 20)] + [i for i in range(20, 200, 10)] + [i for i in range(200, max_iter, 2000)] + [max_iter]

files = glob.glob('../results/train_path_init0*.npz')
data_dict = {}
list_hyperparam = []
for file in tqdm(files):
    names = file.split("_")
    init = int(names[2].strip("init"))
    lr = float(names[3].strip("lr"))
    bs = int(names[4].strip("bs"))
    neurons = int(names[5].strip("neurons"))
    list_hyperparam.append((init, lr, bs, neurons))
    data = np.load(f"../results/train_path_init{init}_lr{lr}_bs{bs}_neurons{neurons}_max_iter200000.npz")["training_path"]
    lk = data[-1]
    p1, e1 = hist(lk[lk >= 0], bins=100)
    p2, e2 = hist(-lk[lk < 0], bins=100)
    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_e1"] = e1
    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_p1"] = p1
    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_e2"] = e2
    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_p2"] = p2
data_dict[f"list_hyperparam"] = list_hyperparam
np.savez("./figures/param_dist.npz", **data_dict)
