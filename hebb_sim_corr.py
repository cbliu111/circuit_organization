from utils import *
from scipy.sparse import csc_matrix
from scipy.optimize import curve_fit
from scipy.interpolate import make_smoothing_spline
import scipy
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def density(W, th):
    W = np.abs(W) > th
    W = W.astype(float)
    N = W.shape[0]
    return np.sum(W) / (N*(N-1))
    
def clustering_coefficient(W, th):
    W = np.abs(W) > th
    W = W.astype(float)
    N = W.shape[0]
    # Degrees:
    ks = np.sum(W, axis=1)
    # Number of triangles for each node:
    triangles = np.diag(W @ np.triu(W) @ W)
    # Number of triplets for each node:
    ks_nontrivial = ks[ks > 1]
    triplets = ks_nontrivial * (ks_nontrivial - 1) / 2
    # Clustering coefficient:
    cc = np.sum(triangles) / (np.sum(triplets)+1e-30)
    return cc

def heterogeneity(W, I, J, th):
    conn_strengths = W[I, J]
    conn_strengths = conn_strengths[conn_strengths > th]
    # Unique strengths and counts:
    s_values, s_counts = np.unique(conn_strengths, return_counts=True)
    # Normalize the counts to sum to 1:
    p = s_counts / (np.sum(s_counts)+1e-30)
    # Compute heterogeneity:
    diff_matrix = np.abs(s_values[:, None] - s_values[None, :])    # Matrix of |s_i - s_j|
    P_matrix = np.outer(p, p)                                      # p_i * p_j
    heterogeneity = 0.5 * np.sum(diff_matrix * P_matrix) / (np.mean(conn_strengths)+1e-30)
    return heterogeneity

def hist(x, bins=100, bd=None):
    if bd:
        p, e = np.histogram(x, bins=bins, range=bd, density=True)
    else:
        p, e = np.histogram(x, bins=bins, density=True)
    idx = p > 0
    p = p[idx]
    # e = 0.5 * (e[:-1] + e[1:])
    e = e[1:]
    e = e[idx]
    p += 1e-15
    p = p / p.sum()
    return p, e


corr = np.load("./figures/init0_lr0.1_bs64_neurons50/corr_mat_all.npz")
ts = np.zeros((len(corr),))
Cs = np.zeros((len(corr), 110, 110))
for i, k in enumerate(corr):
    if "corr" in k:
        t = k.strip("corr_mat")
        ts[i] = t
        Cs[i] = corr[k]

realmin = 1e-30

# architecture
N = 110
E = int(N*(N-1)/2)
I, J = np.triu_indices(N, k=1)
edge_inds = np.ravel_multi_index((I, J), (N, N))

for pp in np.linspace(0.1, 0.9, 20):
    # hyperparameter
    p_Hebb = 0.37
    lr = 1
    p_update = 0.01 / lr
    n_update = int(E * p_update)
    total_num_updates = 200000
    
    # initialization
    # inds_sample = np.random.choice(E, int(0.1*E), replace=True)
    # row = I[inds_sample]
    # col = J[inds_sample]
    # W = csc_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N)).toarray()
    # W = W + W.T  # Symmetrize the matrix
    # corr_func = scipy.interpolate.interp1d(ts, Cs, axis=0)
    # C = corr["corr_mat0"]
    
    W = np.zeros((N, N))
    corr_func = scipy.interpolate.interp1d(ts, Cs, axis=0)
    C = corr["corr_mat0"]
    prob = np.abs(C[I, J])+realmin
    prob /= prob.sum()
    inds_inc = np.random.choice(E, int(0.1/lr*E), replace=True, p=prob)
    row, col = I[inds_inc], J[inds_inc]
    W = W + lr * csc_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N)).toarray()
    
    # record data
    data_dict = {}
    data_dict["W0"] = W
    data_dict["C0"] = C
    
    for t in tqdm(range(1, total_num_updates+1)):
        # pick connections to prune
        # effective prune size should be proportional to strength
        inds_remove = np.random.choice(E, n_update, replace=False)
        row, col = I[inds_remove], J[inds_remove]
        # prune connections with continuous value
        mass = W[row, col]
        W = W - csc_matrix((mass, (row, col)), shape=(N, N)).toarray()
        mass_remove = int(np.ceil(np.sum(mass) / lr))
        
        # pick connections to increase with Hebbian growth
        # effective jump size is automatically determined by the number of selected preferential connections
        prob = np.abs(C[I, J])+realmin
        # prob = np.abs(W[I, J])+realmin
        prob /= prob.sum()
        n_Hebb = np.random.binomial(mass_remove, p_Hebb)
        inds_inc_Hebb = np.random.choice(E, n_Hebb, replace=True, p=prob)
        n_random = mass_remove - n_Hebb
        # print("remove", mass_remove)
        # print("n_random", n_random)
        # print("n_Hebb", n_Hebb)
        # pick connections to increase randomly
        inds_inc_rand = np.random.choice(E, n_random, replace=True)
        inds_inc = np.concatenate([inds_inc_Hebb, inds_inc_rand], axis=0)
        row, col = I[inds_inc], J[inds_inc]
        # print(row.shape, col.shape)
        W = W + lr * csc_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N)).toarray()
        # print(np.sum(W))
        W[J, I] = W[I, J]  # only operate on the upper triangle, lower is symmetry since undirected graph
        if np.any(W < 0):
            print("Negative W", flush=True)
            exit()
        C = corr_func(t)
        if t in ts:
            data_dict[f"W{t}"] = W
            data_dict[f"C{t}"] = C
            # print("reduce mass", np.sum(mass))
            # print("add mass", lr * len(row))
            # print(np.sum(W))
    np.savez(f"./figures/init0_lr0.1_bs64_neurons50/hebb_sim_data_empirical_corr_p{pp}.npz", **data_dict)
    
    n_traj = 100
    W = data_dict[f"W{200000}"]
    Ws = np.zeros((n_traj, W.shape[0], W.shape[1]))
    C = corr["corr_mat200000"]
    for i in tqdm(range(n_traj)):
        for j in range(100):
            inds_remove = np.random.choice(E, n_update, replace=False)
            row, col = I[inds_remove], J[inds_remove]
            # prune connections with continuous value
            mass = W[row, col]
            W = W - csc_matrix((mass, (row, col)), shape=(N, N)).toarray()
            mass_remove = int(np.sum(mass) / lr)
            
            # pick connections to increase with Hebbian growth
            # effective jump size is automatically determined by the number of selected preferential connections
            # prob = np.abs(C[I, J])+realmin
            prob = np.abs(W[I, J])+realmin
            prob /= prob.sum()
            n_Hebb = np.random.binomial(mass_remove, p_Hebb)
            inds_inc_Hebb = np.random.choice(E, n_Hebb, replace=True, p=prob)
            n_random = mass_remove - n_Hebb
            # pick connections to increase randomly
            inds_inc_rand = np.random.choice(E, n_random, replace=True)
            inds_inc = np.concatenate([inds_inc_Hebb, inds_inc_rand], axis=0)
            row, col = I[inds_inc], J[inds_inc]
            W = W + lr * csc_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N)).toarray()
            W[J, I] = W[I, J]  # only operate on the upper triangle, lower is symmetry since undirected graph
            if np.any(W < 0):
                print("Negative W", flush=True)
                exit()
        Ws[i] = W
    np.savez(f"./figures/init0_lr0.1_bs64_neurons50/hebb_sim_empirical_corr_dist_data_p{pp}.npz", Ws=Ws)
