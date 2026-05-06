from utils import * 
from visualizer import NNVisualizer
import os 
from tqdm import tqdm
import glob

init = 0
lr = 0.1
bs = 64
neurons = 50
max_iter = 200000
ss = [i for i in range(0, 20)] + [i for i in range(20, 200, 10)] + [i for i in range(200, max_iter, 2000)] + [max_iter]
nnv = NNVisualizer(max_iter=max_iter, test_point_indices=ss, hyperparam=(init, lr, bs, neurons), num_workers=0)


files = glob.glob('../results/train_path_init0_lr0.1_bs64_neurons*_max_iter200000.npz')
data_dict = {}
list_hyperparam = []
for file in tqdm(files):
    # print(file)
    names = file.split("_")
    init = int(names[2].strip("init"))
    lr = float(names[3].strip("lr"))
    bs = int(names[4].strip("bs"))
    neurons = int(names[5].strip("neurons"))
    list_hyperparam.append((init, lr, bs, neurons))
    nnv.set_hyperparam(init=init, lr=lr, bs=bs, neurons=neurons)
    data = nnv.get_neurons_edges(label="critical edges", data_loader="train")
    # nnv.get_ccg_performance(data, label="critical edges", data_loader="train")
    for ti in range(10):
        data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_task{ti}_neurons"] = data[f"task{ti}_neurons"]
        data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_task{ti}_edges"] = data[f"task{ti}_edges"]
data_dict['list_hyperparam'] = list_hyperparam
np.savez("./figures/ccg_neurons_edges_for_model_size.npz", **data_dict)

files = glob.glob('../results/train_path_init0_*_neurons50_max_iter200000.npz')
data_dict = {}
list_hyperparam = []
for file in tqdm(files):
    # print(file)
    names = file.split("_")
    init = int(names[2].strip("init"))
    lr = float(names[3].strip("lr"))
    if lr >= 1.6:
        continue
    bs = int(names[4].strip("bs"))
    neurons = int(names[5].strip("neurons"))
    list_hyperparam.append((init, lr, bs, neurons))
    nnv.set_hyperparam(init=init, lr=lr, bs=bs, neurons=neurons)
    data = nnv.get_neurons_edges(label="critical edges", data_loader="train")
    # nnv.get_ccg_performance(data, label="critical edges", data_loader="train")
    for ti in range(10):
        data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_task{ti}_neurons"] = data[f"task{ti}_neurons"]
        data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_task{ti}_edges"] = data[f"task{ti}_edges"]
data_dict['list_hyperparam'] = list_hyperparam
np.savez("./figures/ccg_neurons_edges_for_lr_bs.npz", **data_dict)