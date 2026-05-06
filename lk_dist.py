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

dirs = glob.glob("../results/figures/*")
data_dict = {}
list_hyperparam = []
for dir in tqdm(dirs):
    name = dir.split("/")[-1]
    # names = file.split("_")
    names = name.split("_")
    init = int(names[0].strip("init"))
    if init > 0:
        continue
    lr = float(names[1].strip("lr"))
    bs = int(names[2].strip("bs"))
    neurons = int(names[3].strip("neurons"))
    list_hyperparam.append((init, lr, bs, neurons))
    data = np.load(f"../results/figures/init{init}_lr{lr}_bs{bs}_neurons{neurons}/per_data_loss.npz")
    lk = data["iter200000"]
    lk = np.nan_to_num(lk, nan=100)
    p, e = hist(lk, bins=500)
    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_e"] = e
    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_p"] = p
data_dict[f"list_hyperparam"] = list_hyperparam
np.savez("./figures/per_data_loss_dist_hyperparam.npz", **data_dict)
