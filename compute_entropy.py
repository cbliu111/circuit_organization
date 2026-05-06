from utils import *
from glob import glob
from visualizer import NNVisualizer
from tqdm import tqdm
from multiprocessing import Pool

init = 0
lr = 0.1
bs = 64
neurons = 50
max_iter = 200000
ss = [i for i in range(0, 20)] + [i for i in range(20, 200, 10)] + [i for i in range(200, max_iter, 2000)] + [max_iter]
nnv = NNVisualizer(max_iter=max_iter, test_point_indices=ss, hyperparam=(init, lr, bs, neurons), num_workers=0)

def get_entropy(file):
    if not "train_path" in file:
        return file
    names = file.split("_")
    init = int(names[2].strip("init"))
    lr = float(names[3].strip("lr"))
    bs = int(names[4].strip("bs"))
    neurons = int(names[5].strip("neurons"))
    nnv.set_hyperparam(init=init, lr=lr, bs=bs, neurons=neurons)
    nnv.get_entropies()
    return file

if __name__ == '__main__':

    files = glob('../results/train_path_init0*.npz')
    with Pool(10) as p:
        print(p.map(get_entropy, files), flush=True)


# for file in tqdm(files):
#     if not "train_path" in file:
#         continue
#     names = file.split("_")
#     init = int(names[2].strip("init"))
#     lr = float(names[3].strip("lr"))
#     bs = int(names[4].strip("bs"))
#     neurons = int(names[5].strip("neurons"))
#     nnv.set_hyperparam(init=init, lr=lr, bs=bs, neurons=neurons)
#     nnv.get_entropies()

