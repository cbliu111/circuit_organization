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

files = glob.glob('../results/train_path_init0_*.npz')
data_dict = {}
list_hyperparam = []
for file in tqdm(files):
    names = file.split("_")
    init = int(names[2].strip("init"))
    lr = float(names[3].strip("lr"))
    bs = int(names[4].strip("bs"))
    neurons = int(names[5].strip("neurons"))
    list_hyperparam.append((init, lr, bs, neurons))
    data = np.load(f"./figures/init{init}_lr{lr}_bs{bs}_neurons{neurons}/corr_mat.npz")
    corr = np.abs(data["corr"])
    corr = np.nan_to_num(corr, nan=0)
    p, e = hist(corr.reshape(-1), bins=500)
    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_e"] = e
    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_p"] = p
data_dict[f"list_hyperparam"] = list_hyperparam
np.savez("./figures/corr_dist.npz", **data_dict)
