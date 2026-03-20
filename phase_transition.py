from utils import *
import glob
from visualizer import NNVisualizer
from tqdm import tqdm

init = 0
lr = 0.1
bs = 64
neurons = 50
max_iter = 200000
ss = [i for i in range(0, 20)] + [i for i in range(20, 200, 10)] + [i for i in range(200, max_iter, 2000)] + [max_iter]
# nnv = NNVisualizer(max_iter=max_iter, test_point_indices=ss, hyperparam=(init, lr, bs, neurons), num_workers=0)

# files = glob.glob('./autodl-tmp/train_path_init0*.npz')
dirs = glob.glob("./autodl-tmp/figures/*")
data_dict = {}
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
    # nnv.set_hyperparam(init=init, lr=lr, bs=bs, neurons=neurons)
    max_cluster_sizes = []
    average_degrees = []
    average_finite_cluster_sizes = []
    frobenius_norms = []
    susc = []
    for s in ss:
        data = np.load(f"{dir}/activity_key_measures_iter{s}.npz")
        # data = nnv.get_activity_key_measures(s=s)
        frobenius_norms.append(data['frobenius_norm'])
        sizes = np.array(data['cluster_sizes_cth0.5'])
        max_cluster_sizes.append(np.max(sizes))
        degrees = np.array(data['degree_cth0.5'])
        if len(degrees) == 0:
            average_degrees.append(0)
        else:
            average_degrees.append(np.mean(degrees))
        sizes_excl_gcc = sizes[sizes < sizes.max()]
        if len(sizes_excl_gcc) == 0:
            average_finite_cluster_sizes.append(0)
            susceptibility = 0
        else:
            susceptibility = (np.mean(sizes_excl_gcc**2) - np.mean(sizes_excl_gcc)**2) / (np.mean(sizes_excl_gcc) + 1e-12)
            average_finite_cluster_sizes.append(np.mean(sizes_excl_gcc))
        susc.append(susceptibility)

    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_frobenius_norm"] = frobenius_norms
    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_max_cluster_size"] = max_cluster_sizes
    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_average_degree"] = average_degrees
    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_average_finite_cluster_sizes"] = average_finite_cluster_sizes
    data_dict[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_susceptibility"] = susc
np.savez("./figures/phase_transition.npz", **data_dict)