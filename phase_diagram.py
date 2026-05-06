from utils import *
from glob import glob
from visualizer import NNVisualizer
from tqdm import tqdm

init = 0
lr = 0.1
bs = 64
neurons = 50
max_iter = 200000
ss = [i for i in range(0, 20)] + [i for i in range(20, 200, 10)] + [i for i in range(200, max_iter, 2000)] + [max_iter]
nnv = NNVisualizer(max_iter=max_iter, test_point_indices=ss, hyperparam=(init, lr, bs, neurons), num_workers=0)

files = glob('../results/*.npz')
list_hyperparam = []
data_dict = {}
for file in files:
    if not "train_path_init0" in file:
        continue
    names = file.split("_")
    init = int(names[2].strip("init"))
    lr = float(names[3].strip("lr"))
    bs = int(names[4].strip("bs"))
    neurons = int(names[5].strip("neurons"))
    nnv.set_hyperparam(init=init, lr=lr, bs=bs, neurons=neurons)
    list_hyperparam.append((init, lr, bs, neurons))
    data = np.load(file)
    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_loss"] = data["losses"][-1]
    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_acc"] = data["accuracy"][-1]
    data = nnv.get_per_data_loss()
    lk = data['iter200000']
    l_bar = np.mean(lk)
    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_l_bar"] = l_bar
    Bs = np.linspace(np.mean(lk), np.max(lk), 100)
    entropy = []
    temp = []
    for b in Bs:
        t = get_temperature(lk, b)
        e = quasi_entropy(lk, t)
        entropy.append(e)
        temp.append(t)
    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_entropy"] = entropy
    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_temp"] = temp
    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_Bs"] = Bs
data_dict["list_hyperparam"] = list_hyperparam
np.savez(f"./figures/loss_acc_entropy.npz", **data_dict)