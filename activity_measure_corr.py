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

init = 0
lr = 0.1
bs = 64
model_sizes = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]
for m in model_sizes:
    nnv.set_hyperparam(init=init, lr=lr, bs=bs, neurons=m)
    for s in ss:
        nnv.get_activity_measures(s=s)
        try:
            os.remove(f"./figures/init{init}_lr{lr}_bs{bs}_neurons{m}/activity_iter{s}.npz")
        except:
            pass
        try:
            os.remove(f"./figures/init{init}_lr{lr}_bs{bs}_neurons{m}/spks_iter{s}.npz")
        except:
            pass

files = glob.glob('../results/*.npz')
list_hyperparam = []
data_dict = {}
for file in tqdm(files):
    if not "train_path_init0" in file:
        continue
    names = file.split("_")
    init = int(names[2].strip("init"))
    lr = float(names[3].strip("lr"))
    bs = int(names[4].strip("bs"))
    m = int(names[5].strip("neurons"))
    list_hyperparam.append((init, lr, bs, m))
    save_name = f"./figures/init{init}_lr{lr}_bs{bs}_neurons{m}/corr_mat.npz"
    
    if os.path.exists(save_name):
        continue

    data = np.load(file)
    nnv.set_hyperparam(init=init, lr=lr, bs=bs, neurons=m)
    s = 200000
    spks = nnv.get_spks(s=s)['spks']
    corr = np_pearson_corr(spks, spks)
    np.savez(save_name, corr=corr)
    try:
        os.remove(f"./figures/init{init}_lr{lr}_bs{bs}_neurons{m}/activity_iter0.npz")
    except:
        pass
    try:
        os.remove(f"./figures/init{init}_lr{lr}_bs{bs}_neurons{m}/spks_iter0.npz")
    except:
        pass
    try:
        os.remove(f"./figures/init{init}_lr{lr}_bs{bs}_neurons{m}/activity_iter{s}.npz")
    except:
        pass
    try:
        os.remove(f"./figures/init{init}_lr{lr}_bs{bs}_neurons{m}/spks_iter{s}.npz")
    except:
        pass