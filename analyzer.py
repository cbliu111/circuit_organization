import copy
import itertools
import os
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from rastermap.sort import traveling_salesman
import numpy as np
from numpy.polynomial.laguerre import lagtrim
from torch.func import jacrev, vmap, functional_call
from utils import *
import scipy
from scipy.sparse import csc_matrix
import networkx as nx
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD


class NNAnalyzer:
    """
    compute_hessian_block is used for a 1 hidden layer fully-connected network, other type of network should
    implement the forward computation in this function.
    Assume criterion use reduction type 'none'.
    Only support pytorch dataloader.
    """

    def __init__(self,
                 num_tasks=10,
                 device='cpu',
                 data_folder="./autodl-tmp/",
                 save_folder="./figures/"
                 ):
        self.num_tasks = num_tasks
        self.model = None
        self.device = device
        self.train_loader = None
        self.test_loader = None
        self.criterion = None
        self.params = None  # model parameters
        self.data, self.target = None, None
        self.test_data, self.test_target = None, None
        self.mlp = None  # original model
        self.data_folder = data_folder
        self.prefix = save_folder
        self.critical_connections = [0, 1]
        self.non_critical_connections = [0, 1]
        self.max_iter = None
        self.test_point_indices = None
        self.cth = None

        self.path_data = None
        self.hyperparam = None
        self.save_folder = None
        self.path_file = None
        self.neuron_pos = None
        self.num_workers = 0

    def set_hyperparam(self, init=0, lr=0.1, bs=64, neurons=50):
        self.hyperparam = (init, lr, bs, neurons)
        init, lr, bs, neurons = self.hyperparam
        n = neurons
        self.neuron_pos = [(0, i) for i in range(n)] + [(1, i) for i in range(n)] + [(2, i) for i in range(self.num_tasks)]
        self.save_folder = self.prefix + f"init{init}_lr{lr}_bs{bs}_neurons{neurons}/"
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
        self.path_file = self.data_folder + f"train_path_init{init}_lr{lr}_bs{bs}_neurons{neurons}_max_iter{self.max_iter}.npz"
        self.path_data = np.load(self.path_file)["training_path"]

    def init_dataset(self):
        train_dataset = torchvision.datasets.MNIST(
            root='./MNIST',
            train=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
            download=True
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./MNIST',
            train=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
            download=True
        )
        self.train_loader = DataLoader(train_dataset, batch_size=60000, shuffle=True, num_workers=self.num_workers,
                                       pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=60000, shuffle=False, num_workers=self.num_workers,
                                      pin_memory=True)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.data, self.target = next(iter(self.train_loader))
        self.data, self.target = self.data.to(self.device), self.target.to(self.device)
        self.test_data, self.test_target = next(iter(self.test_loader))
        self.test_data, self.test_target = self.test_data.to(self.device), self.test_target.to(self.device)
        # for self.data, self.target in self.train_loader:
        #     self.data, self.target = self.data.to(self.device), self.target.to(self.device)

    def init_model(self):
        init, lr, bs, neurons = self.hyperparam
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, 10)
        )
        self.mlp.to(self.device)
        self.mlp.eval()
        s = self.max_iter
        test_point_indices = self.test_point_indices
        path = self.path_data
        with torch.no_grad():
            path_index = np.where(np.array(test_point_indices) == s)[0]
            weights = restore_weights(path[path_index.item()], neurons)
            for i, w in enumerate(self.mlp.parameters()):
                w.data = torch.tensor(weights[i]).to(self.device)
        self.model = copy.deepcopy(self.mlp)
        self.model.eval()
        self.params = [p for p in self.model.parameters() if p.requires_grad]

    def update_model(self, s=0):
        self.init_model()
        assert s in self.test_point_indices, f"Iteration {s} not recorded in {self.test_point_indices}."
        test_point_indices = self.test_point_indices
        init, lr, bs, neurons = self.hyperparam
        path = self.path_data
        self.model = copy.deepcopy(self.mlp)
        with torch.no_grad():
            path_index = np.where(np.array(test_point_indices) == s)[0]
            weights = restore_weights(path[path_index.item()], neurons)
            for i, w in enumerate(self.model.parameters()):
                w.data = torch.tensor(weights[i]).to(self.device)
        self.model.to(self.device)
        self.model.eval()
        self.params = [p for p in self.model.parameters() if p.requires_grad]

    def info(self, message):
        init, lr, bs, neurons = self.hyperparam
        print(f"Init {init}, lr {lr}, bs {bs}, neurons {neurons}, {message} finished.")

    def warn(self, message):
        init, lr, bs, neurons = self.hyperparam
        print(f"Init {init}, lr {lr}, bs {bs}, neurons {neurons}, {message}.")

    def get_activity(self, s=0, only_correct=False):
        file = self.save_folder + f"activity_iter{s}.npz"
        if os.path.exists(file):
            return np.load(file)

        self.update_model(s=s)
        model = copy.deepcopy(self.model)
        model.eval()
        feature_size = []

        def hook(module, in_data, out_data):
            module_name.append(module.__class__)
            in_hook.append(in_data[0].clone().detach())
            feature_size.append(in_data[0].shape[1])
            od = out_data.clone().detach()
            out_hook.append(od)
            feature_size.append(out_data.shape[1])
            return None

        for child in model.children():
            child.register_forward_hook(hook=hook)

        data_dict = {}
        for label in range(self.num_tasks):
            module_name = []
            in_hook = []
            out_hook = []

            idx = torch.where(self.target == label)
            data, target = self.data[idx], self.target[idx]
            output = model(data)
            loss = self.criterion(output, target).mean()  # for using none reduction
            pred = output.data.max(1)[1]
            correct = pred.eq(target).sum()
            # print(f"loss: {loss}, correct percent: {correct / len(target) * 100} %")

            correct_idx = pred == target
            # activity = torch.concat([in_hook[0]] + out_hook, dim=1).permute(1, 0)
            activity = torch.cat(out_hook, dim=1)
            # correct_activity = activity[:, correct_idx]  # dim 0 is list of activities, dim 1 is input data
            if only_correct:
                correct_activity = activity[correct_idx]
            else:
                correct_activity = activity
            data_dict[f"label_{label}"] = correct_activity.detach().cpu().numpy()
            # save the original activity data
            torch.cuda.empty_cache()

        data_dict["hyperparam"] = self.hyperparam
        np.savez(file, **data_dict)
        return data_dict

    def get_spks(self, s=0, only_correct=False):
        file = self.save_folder + f"spks_iter{s}.npz"
        if os.path.exists(file):
            return np.load(file)

        data = self.get_activity(s=s)
        init, lr, bs, neurons = self.hyperparam
        raw_activity_collections = []
        activity_collections = []
        labels = []
        num_data = []
        n = neurons
        for i in range(self.num_tasks):
            activity = np.transpose(data["label_" + str(i)], (1, 0))
            show_idx = np.concatenate([
                np.arange(n) + 784 + n,
                np.arange(n) + 784 + 3 * n,
                np.arange(self.num_tasks) + 784 + 4 * n], axis=0)
            raw_d2 = activity[show_idx, :]
            raw_activity_collections.append(raw_d2)
            # scale across data is helpful when the dataset is very much diverse
            # d2 = linear_scale_data(d2, dim=1)
            # z-score across all the activities across layers
            d2 = scipy.stats.zscore(raw_d2, axis=0)
            # the outputs associated with a constant input is set to be nan in scipy.stats.zscore
            # reset the activity to 0
            raw_d2[np.isnan(raw_d2)] = 0
            d2[np.isnan(d2)] = 0
            num_data.append(d2.shape[1])
            activity_collections.append(d2)
            labels.append(i * np.ones(activity.shape[1]))
        spks = np.concatenate(activity_collections, axis=1)
        raw_spks = np.concatenate(raw_activity_collections, axis=1)
        mean_raw_spks = np.mean(raw_spks, axis=1)
        data_dict = {
            'spks': spks,
            'labels': np.concatenate(labels, axis=0),
            'num_data': num_data,
            'raw': raw_spks,
            'mean_raw_spks': mean_raw_spks,
            "hyperparam": self.hyperparam
        }
        np.savez(file, **data_dict)
        return data_dict

    def get_activity_measures_minibatch(
            self,
            s=0,
            bs=64,
            num_minibatch=10000,
    ):
        file = self.save_folder + f"activity_measures_minibatch_iter{s}.npz"
        if os.path.exists(file):
            return np.load(file)

        ts = self.cth
        file = self.save_folder + f"activity_iter{s}.npz"
        if not os.path.exists(file):
            self.get_activity(s=s)
        data = np.load(file)
        data_dict = {"correlation_thresholds": ts,
                     "number_of_minibatch": num_minibatch,
                     "minibatch_size": bs}
        spks = data['spks']
        mean_raw_spks = data["mean_raw_spks"]
        for icount, t in enumerate(ts):
            # distribution of cluster sizes
            data_dict[f"connectivity_prob_cth{t}"] = []
            data_dict[f"number_of_edges_cth{t}"] = []
            data_dict[f"cluster_sizes_cth{t}"] = []
            data_dict[f"cluster_values_cth{t}"] = []
            data_dict[f"clique_sizes_cth{t}"] = []
            data_dict[f"clique_values_cth{t}"] = []
            for _ in range(num_minibatch):
                indices = np.random.randint(0, spks.shape[1], size=(bs,))  # minibatch size 32 is the same as the training
                minibatch = spks[:, indices]
                corr = np_pearson_corr(minibatch, minibatch)
                conn_graph = np.abs(corr) >= t
                data_dict[f"connectivity_prob_cth{t}"].append(np.sum(conn_graph) / (corr.shape[0] * corr.shape[1]))
                data_dict[f"number_of_edges_cth{t}"].append(np.sum(conn_graph))
                # create graph
                G = nx.Graph()
                n = corr.shape[0]
                G.add_nodes_from(range(n))
                for i in range(n):
                    for j in range(i + 1, n):  # Only upper triangle to avoid duplicates
                        if np.abs(corr[i, j]) >= t:  # Elements are connected
                            G.add_edge(i, j, weight=corr[i, j])
                clusters = list(nx.connected_components(G))
                for _, c in enumerate(clusters):
                    data_dict[f"cluster_sizes_cth{t}"].append(len(c))
                    idx = np.array(list(c))
                    data_dict[f"cluster_values_cth{t}"].append(np.sum(np.abs(mean_raw_spks[idx])))
                cliques = nx.find_cliques(G)
                for _, c in enumerate(cliques):
                    data_dict[f"clique_sizes_cth{t}"].append(len(c))
                    idx = np.array(list(c))
                    data_dict[f"clique_values_cth{t}"].append(np.sum(np.abs(mean_raw_spks[idx])))
        data_dict["hyperparam"] = self.hyperparam
        np.savez(file, **data_dict)
        return data_dict

    def get_activity_key_measures(
            self,
            s=0,
            threshold_method="hard",  # "weighted" or "proportional"
            treat_negatives="abs",  # "zero" or "abs" or "keep"
            use_fisher_z=True,
            # densities=np.linspace(0.05, 0.25, 9),
            sigma_comm=0.5,
            avalanche_thr=0.3,
            kuramoto_delta=1.0,
            mu_over_beta=1.0,
            random_seed=0,
            cth=0.5,
            frac=0.15,
            keep_activity_files=False,
    ):
        file = self.save_folder + f"activity_key_measures_iter{s}.npz"
        if os.path.exists(file):
            return np.load(file)

        data = self.get_spks(s=s)
        data_dict = {
            'iteration': s,
            'threshold_method': threshold_method,
            'treat_negative': treat_negatives,
            'use_fisher_z': use_fisher_z,
            'avalanche_thr': avalanche_thr,
            'kuramoto_delta': kuramoto_delta,
            'mu_over_beta': mu_over_beta,
            'random_seed': random_seed,
            'correlation_threshold': cth,
            'keep_weight_fraction': frac,
        }

        spks = data["spks"]
        mean_raw_spks = data["mean_raw_spks"]
        # obtain functional connectome
        corr = np_pearson_corr(spks, spks)
        # frobenius norm
        data_dict["frobenius_norm"] = np.sum(corr ** 2)

        # preprocessing
        corr = symmetrize(corr)
        corr = zero_diagonal(corr)
        if use_fisher_z:
            corr = fisher_z_transform(corr)
        if treat_negatives == "zero":
            corr = keep_nonnegative(corr)
        elif treat_negatives == "abs":
            corr = np.abs(corr)
        elif treat_negatives == "keep":
            pass
        else:
            raise ValueError("treat_negatives must be 'zero', 'abs', or 'keep'")

        # binarize
        if threshold_method == "hard" and cth:
            Ath = hard_threshold(corr, cth)
        elif threshold_method == "weighted":
            Ath = weighted_threshold(corr, frac)
        elif threshold_method == "proportional":
            Ath = proportional_threshold(corr, frac)
        else:
            raise ValueError("treat_negatives must be 'hard', 'weighted', or 'proportional'")

        Abin = (Ath > 0).astype(int)
        G = nx.from_numpy_array(Abin)
        largest_cc_nodes = max(nx.connected_components(G), key=len)
        Glcc = G.subgraph(largest_cc_nodes).copy()
        nodes = sorted(Glcc.nodes())
        Alcc = nx.to_numpy_array(Glcc, nodelist=nodes, dtype=int)

        S1 = largest_component_fraction(Abin)
        _, sus = component_size_stats(Abin)
        Eglob = global_efficiency(Ath, weighted=True)
        rho = spectral_radius(Ath)
        lam2 = algebraic_connectivity(Alcc)
        kmax = kcore_max_k(Abin)
        sis_thr, sis_indicator = sis_threshold(Ath, mu_over_beta=mu_over_beta)
        Kc = kuramoto_Kc(Alcc, delta=kuramoto_delta)
        tau_mix = diffusion_mixing_time(Alcc)
        cc = clustering_coefficient(Abin)
        heter = heterogeneity(corr)
        branching = branching_ratio_proxy(symmetrize(corr), thr=avalanche_thr)
        sigma, omega = small_worldness(Glcc)
        # try:
        #     sigma = nx.sigma(Glcc, niter=10, nrand=20, seed=random_seed)
        # except ZeroDivisionError:
        #     sigma = np.nan
        # try:
        #     omega = nx.omega(Glcc, niter=10, nrand=20, seed=random_seed)
        # except ZeroDivisionError:
        #     omega = np.nan

        data_dict["largest_component_fraction"] = S1
        data_dict["susceptibility"] = sus
        data_dict["global_efficiency"] = Eglob
        data_dict["small_worldness_sigma"] = sigma
        data_dict["small_worldness_omega"] = omega
        data_dict["clustering_coefficient"] = cc
        data_dict["heterogeneity"] = heter
        data_dict["spectral_radius"] = rho
        data_dict["algebraic_connectivity"] = lam2
        data_dict["kcore_max_k"] = kmax
        data_dict["SIS_indicator"] = sis_indicator
        data_dict["kuramoto_Kc"] = Kc
        data_dict["diffusion_mixing_time"] = tau_mix
        data_dict["branching_ratio_proxy"] = branching

        # connectivity probability
        data_dict[f"connectivity_prob_cth{cth}"] = np.sum(Abin) / (corr.shape[0] * corr.shape[1])
        # edges
        data_dict[f"number_of_edges_cth{cth}"] = np.sum(Abin)

        # create graph
        G = nx.Graph()
        n = corr.shape[0]
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):  # Only upper triangle to avoid duplicates
                if corr[i, j] >= cth:  # Elements are connected
                    G.add_edge(i, j, weight=corr[i, j])
        degrees = [d for _, d in G.degree()]
        data_dict[f"degree_cth{cth}"] = degrees

        # Forman-Ricci entropy
        curvatures = compute_forman_ricci_curvatures(G)
        curvature_values = list(curvatures.values())
        # Shift curvatures to make them all positive (for probability distribution)
        if len(curvature_values) > 0:
            min_curvature = min(curvature_values)
            if min_curvature < 0:
                shifted_curvatures = [c - min_curvature + 0.1 for c in curvature_values]
            else:
                shifted_curvatures = [c + 0.1 for c in curvature_values]
                # Create probability distribution
            total = sum(shifted_curvatures)
            probabilities = [c / total for c in shifted_curvatures]
            # Calculate Shannon entropy
            entropy = -sum(p * math.log(p) for p in probabilities)
            data_dict[f"forman_ricci_entropy_cth{cth}"] = entropy
            data_dict[f"forman_ricci_curvatures_cth{cth}"] = curvatures
        else:
            data_dict[f"forman_ricci_entropy_cth{cth}"] = 0
            data_dict[f"forman_ricci_curvatures_cth{cth}"] = 0

        # cluster size
        clusters = list(nx.connected_components(G))
        data_dict[f"cluster_sizes_cth{cth}"] = []
        data_dict[f"cluster_values_cth{cth}"] = []
        for _, c in enumerate(clusters):
            data_dict[f"cluster_sizes_cth{cth}"].append(len(c))
            idx = np.array(list(c))
            data_dict[f"cluster_values_cth{cth}"].append(np.sum(np.abs(mean_raw_spks[idx])))

        # clique size
        cliques = nx.find_cliques(G)
        data_dict[f"clique_sizes_cth{cth}"] = []
        data_dict[f"clique_values_cth{cth}"] = []
        for _, c in enumerate(cliques):
            data_dict[f"clique_sizes_cth{cth}"].append(len(c))
            idx = np.array(list(c))
            data_dict[f"clique_values_cth{cth}"].append(np.sum(np.abs(mean_raw_spks[idx])))

        data_dict["hyperparam"] = self.hyperparam
        np.savez(file, **data_dict)
        if not keep_activity_files:
            try:
                os.remove(self.save_folder + f"activity_iter{s}.npz")
            except FileNotFoundError:
                print("File not exists.")
            try:
                os.remove(self.save_folder + f"spks_iter{s}.npz")
            except FileNotFoundError:
                print("File not exists.")
        return data_dict

    def get_activity_measures(
            self,
            s=0,
    ):
        file = self.save_folder + f"activity_measures_iter{s}.npz"
        if os.path.exists(file):
            return np.load(file)

        ts = self.cth
        data = self.get_spks(s=s)
        data_dict = {"correlation_thresholds": ts}

        spks = data["spks"]
        mean_raw_spks = data["mean_raw_spks"]
        # svd, spectral analysis
        n_pc = min(spks.shape[0], 1000)
        pc_indices = np.arange(n_pc) + 1
        svd = TruncatedSVD(n_components=n_pc).fit(spks.T)  # the maximum number of components are the number of neurons
        svd_vectors = svd.transform(spks.T)
        cumulative_explained_variance = svd.explained_variance_ratio_.cumsum()
        eigenvalues = svd.singular_values_
        normalized_eigenvalues = eigenvalues / eigenvalues.sum()
        end_eign = min(eigenvalues.shape[0], 50)
        alpha, ypred = get_powerlaw(normalized_eigenvalues, np.arange(4, end_eign).astype(int))
        data_dict["cumulative_explained_variance"] = cumulative_explained_variance
        data_dict["pc_indices"] = pc_indices
        data_dict["svd_trans_mat"] = svd.components_
        data_dict["normalized_eigenvalues"] = normalized_eigenvalues
        data_dict["svd_fit_range"] = (7, 110)
        data_dict["svd_fit_params"] = alpha
        # correlation matrix
        corr = np_pearson_corr(spks, spks)
        # frobenius norm
        data_dict["frobenius_norm"] = np.sum(corr ** 2)

        # screen correlation threshold
        for icount, t in enumerate(ts):
            conn_graph = np.abs(corr) > t
            # connectivity probability
            data_dict[f"connectivity_prob_cth{t}"] = np.sum(conn_graph) / (corr.shape[0] * corr.shape[1])
            # edges
            data_dict[f"number_of_edges_cth{t}"] = np.sum(conn_graph)

            # create graph
            G = nx.Graph()
            n = corr.shape[0]
            G.add_nodes_from(range(n))
            for i in range(n):
                for j in range(i + 1, n):  # Only upper triangle to avoid duplicates
                    if corr[i, j] >= t:  # Elements are connected
                        G.add_edge(i, j, weight=corr[i, j])
            degrees = [d for _, d in G.degree()]
            data_dict[f"degree_cth{t}"] = degrees

            # Forman-Ricci entropy
            curvatures = compute_forman_ricci_curvatures(G)
            curvature_values = list(curvatures.values())
            # Shift curvatures to make them all positive (for probability distribution)
            if len(curvature_values) > 0:
                min_curvature = min(curvature_values)
                if min_curvature < 0:
                    shifted_curvatures = [c - min_curvature + 0.1 for c in curvature_values]
                else:
                    shifted_curvatures = [c + 0.1 for c in curvature_values]
                    # Create probability distribution
                total = sum(shifted_curvatures)
                probabilities = [c / total for c in shifted_curvatures]
                # Calculate Shannon entropy
                entropy = -sum(p * math.log(p) for p in probabilities)
                data_dict[f"forman_ricci_entropy_cth{t}"] = entropy
                data_dict[f"forman_ricci_curvatures_cth{t}"] = curvatures
            else:
                data_dict[f"forman_ricci_entropy_cth{t}"] = 0
                data_dict[f"forman_ricci_curvatures_cth{t}"] = 0

            # cluster size
            clusters = list(nx.connected_components(G))
            data_dict[f"cluster_sizes_cth{t}"] = []
            data_dict[f"cluster_values_cth{t}"] = []
            for _, c in enumerate(clusters):
                data_dict[f"cluster_sizes_cth{t}"].append(len(c))
                idx = np.array(list(c))
                data_dict[f"cluster_values_cth{t}"].append(np.sum(np.abs(mean_raw_spks[idx])))

            # clique size
            cliques = nx.find_cliques(G)
            data_dict[f"clique_sizes_cth{t}"] = []
            data_dict[f"clique_values_cth{t}"] = []
            for _, c in enumerate(cliques):
                data_dict[f"clique_sizes_cth{t}"].append(len(c))
                idx = np.array(list(c))
                data_dict[f"clique_values_cth{t}"].append(np.sum(np.abs(mean_raw_spks[idx])))

        # raw activities
        data = self.get_activity(s=s)
        activity_collections = []
        _, _, _, n = self.hyperparam
        for i in range(self.num_tasks):
            activity = np.transpose(data["label_" + str(i)], (1, 0))
            show_idx = np.concatenate([
                np.arange(n) + 784 + n,
                np.arange(n) + 784 + 3 * n,
            ], axis=0)
            d2 = activity[show_idx, :]
            d2[np.isnan(d2)] = 0
            activity_collections.append(d2)
        spks = np.concatenate(activity_collections, axis=1)
        p, e = np.histogram(spks.reshape(-1), bins=100, density=True)
        data_dict["hist_prob"] = p / p.sum()
        data_dict["hist_value"] = e[1:]

        activity_collections = []
        for i in range(self.num_tasks):
            activity = np.transpose(data["label_" + str(i)], (1, 0))
            show_idx = np.concatenate([
                np.arange(self.num_tasks) + 784 + 4 * n,
            ], axis=0)
            d2 = activity[show_idx, :]
            d2[np.isnan(d2)] = 0
            activity_collections.append(d2)
        spks = np.concatenate(activity_collections, axis=1)
        spks = scipy.special.softmax(spks, axis=0)
        p, e = np.histogram(spks.reshape(-1), bins=100, density=True)
        p += 1e-10
        p = p / np.sum(p)
        entropy = np.sum(-p * np.log(p))
        data_dict["pred_hist_prob"] = p
        data_dict["pred_hist_value"] = e[1:]
        data_dict["pred_entropy"] = entropy
        data_dict["hyperparam"] = self.hyperparam
        np.savez(file, **data_dict)
        return data_dict

    def get_grad_cov_parallel(
            self,
            s=0,
            batch_size=64,
            num_samples_cov=int(1e5),
            chunk_size=20,
            exclude_first_layer=False,
            save_cov=False,
            save_sm=False,
            save_gm=False,
            save_gm_mean_var=False,
            random_data=False,
            random_target=False,
    ):
        """As consistent with the hessian computation, one column of cov
        corresponds to <\theta_i \theta_j> for all j"""
        file = self.save_folder + f"cov_iter{s}.npz"
        if os.path.exists(file):
            return np.load(file)

        self.update_model(s=s)
        self.model.train()

        def get_grad(w, x, y):
            o = functional_call(self.model, w, x)
            return self.criterion(o, y)  # Compute loss for the minibatch

        params = {}
        for name, param in self.model.named_parameters():
            if exclude_first_layer:
                if '1' in name:  # first layer should have a prefix including str 1 as label
                    continue
            params[name] = param
        # params = dict(self.model.named_parameters())
        gm = []
        tasks = []
        images, labels = self.data, self.target
        if random_data:
            images = torch.rand_like(images)
        if random_target:
            labels = torch.randint_like(labels, low=labels.min(), high=labels.max())
        images, labels = images.to(self.device), labels.to(self.device).reshape(-1, 1)
        # use Jacobin-forward to gather the gradients
        gs = vmap(
            jacrev(get_grad, argnums=(0)), in_dims=(None, 0, 0), randomness="different", chunk_size=chunk_size,
        )(params, images, labels)
        g = [v.detach().cpu().numpy().reshape(images.shape[0], -1) for _, v in gs.items()]
        g = np.concatenate(g, axis=1)
        gm.append(g)
        tasks.append(labels.detach().cpu().numpy())
        gm = np.concatenate(gm, axis=0)
        gm_mean = np.mean(gm, axis=0)
        gm_var = np.var(gm, axis=0)
        tasks = np.concatenate(tasks)
        del g, gs
        if save_cov:
            num_chunks = num_samples_cov // chunk_size
            noise = np.zeros((num_chunks * chunk_size, gm.shape[1]))
            for i in range(num_chunks):
                indices = np.random.randint(0, gm.shape[0], size=(chunk_size, batch_size))
                # Calculate all minibatch means (may require significant memory for large values)
                minibatch_means = np.array([np.mean(gm[idx], axis=0) for idx in indices])
                # Calculate all noise values in one operation
                noise[i * chunk_size: (i + 1) * chunk_size, :] = gm_mean - minibatch_means
            # pool = mp.Pool(processes=mp.cpu_count())
            # results = pool.map(self.get_noise, [batch_size for _ in range(num_samples_cov)])
            # noise = np.stack(results)
            # cov = np.einsum("ij,ik->jk", noise, noise) / (num_samples_cov - 1)  # divide by total number of samples to normalize
            cov = np.dot(noise.T, noise) / (num_samples_cov - 1)
        else:
            cov = []
        # sm = np.einsum("ij,ik->jk", gm, gm) / gm.shape[0]
        if save_sm:
            sm = np.dot(gm.T, gm) / gm.shape[0]
        else:
            sm = []
        if not save_gm:
            gm = []
            tasks = []
        if not save_gm_mean_var:
            gm_mean = []
            gm_var = []
        data_dict = {
            'cov': cov,
            'sm': sm,
            'gm': gm,
            'gm_mean': gm_mean,
            'gm_var': gm_var,
            'tasks': tasks,
            'hyperparam': self.hyperparam,
        }
        np.savez(file, **data_dict)
        torch.cuda.empty_cache()
        return data_dict

    def get_hessian_block(
            self,
            s=0,
            block_size=int(1e3),
            exclude_first_layer=False,
            random_data=False,
            random_target=False,
    ):
        """compute the full hessian matrix by dividing the matrix in to small blocks,
        each block has shape (block_size, num_params)
        """
        file = self.save_folder + f"hessian_iter{s}.npz"
        if os.path.exists(file):
            return np.load(file)

        def hvp(g, v, p):
            """
            Compute the Hessian-vector product for a given vector `v`.
            """
            hvp_result = torch.autograd.grad(g, p, grad_outputs=v, retain_graph=True)
            return torch.cat([h.view(-1) for h in hvp_result])

        self.update_model(s=s)
        params = [p for p in self.model.parameters() if p.requires_grad]
        total_params = np.sum([p.numel() for p in params])
        block_indices = []
        blocks = total_params // block_size
        if total_params % block_size > 0:
            blocks += 1
        start_index = 0
        for i in range(blocks):
            end_index = min(start_index + block_size, total_params)
            block_indices.append((start_index, end_index))
            start_index = end_index
        hessian_blocks = []
        for i, (start_i, end_i) in enumerate(block_indices):
            numel = min(block_size, end_i - start_i)
            hessian_blocks.append(torch.zeros((numel, total_params), device='cpu'))

        flat_params = flatten_params(params)
        unflattened_params = unflatten_params(flat_params, params)
        # identity_mat = torch.eye(block_size, device=self.device)
        x, y = self.data, self.target
        if random_data:
            x = torch.rand_like(x)
        if random_target:
            y = torch.randint_like(y, low=y.min(), high=y.max())
        x, y = x.to(self.device), y.to(self.device)
        unflattened_params = unflatten_params(flat_params, params)
        # 3 layer mlp in function form
        y_hat = x.reshape(x.shape[0], -1) @ unflattened_params[0].T + unflattened_params[1]
        y_hat = F.relu(y_hat)
        y_hat = y_hat @ unflattened_params[2].T + unflattened_params[3]
        y_hat = F.relu(y_hat)
        y_hat = y_hat @ unflattened_params[4].T + unflattened_params[5]
        hessian_loss = self.criterion(y_hat, y).mean()
        grads = torch.autograd.grad(hessian_loss, unflattened_params, create_graph=True)
        grads = flatten_params(grads)
        for i, (start_i, end_i) in enumerate(block_indices):
            # Batch of row vectors (one-hot vectors for specific rows, or random vectors)
            numel = min(block_size, end_i - start_i)
            row_vectors = torch.zeros((numel, total_params), device=self.device)
            row_vectors[:, start_i:end_i] = torch.eye(numel, device=self.device)  # One-hot vectors for selecting the rows
            hvp_block = torch.func.vmap(hvp, (None, 0, None))(grads, row_vectors, flat_params)
            hessian_blocks[i] += hvp_block.detach().cpu().numpy()
        del grads
        del hvp_block
        torch.cuda.empty_cache()
        hessian = np.concatenate(hessian_blocks, axis=0)
        data_dict = {'hessian': hessian, 'hyperparam': self.hyperparam}
        np.savez(file, **data_dict)
        return data_dict

    def get_critical_connections(self, cth=0.5, s=0):
        file = self.save_folder + f"critical_connections_iter{s}_cth{cth}.npz"
        if os.path.exists(file):
            return np.load(file)

        data = self.get_spks(s=s)
        spks = data['spks']
        num_data = data['num_data']
        corr = np_pearson_corr(spks, spks)
        th = cth  # 0.7 is considered highly correlated
        ix, iy = np.where(np.abs(corr) > th)
        init, lr, bs, neurons = self.hyperparam
        n = neurons
        neuron_pos = [(0, i) for i in range(n)] + [(1, i) for i in range(n)] + [(2, i) for i in range(self.num_tasks)]
        params = [p.detach().cpu().numpy() for p in self.model.parameters() if p.requires_grad]
        pp = [np.zeros_like(x) for x in params]
        for x, y in zip(ix, iy):
            npx, npy = neuron_pos[x], neuron_pos[y]
            if npx[0] != npy[0]:
                if np.abs(npx[0] - npy[0]) > 1:
                    continue
                if npx[0] > npy[0]:
                    npx, npy = npy, npx
                if npx[0] == 0:
                    pp[2][npy[1], npx[1]] = 1
                if npx[0] == 1:
                    pp[4][npy[1], npx[1]] = 1
            else:
                if npx[0] == 0:
                    for i in range(n):
                        npz = (1, i)
                        vx = corr[neuron_pos.index(npx), neuron_pos.index(npz)]
                        vy = corr[neuron_pos.index(npy), neuron_pos.index(npz)]
                        if np.abs(vx) > th and np.abs(vy) > th:
                            pp[2][npz[1], npx[1]] = 1
                            pp[2][npz[1], npy[1]] = 1
                if npx[0] == 1:
                    for i in range(self.num_tasks):
                        npz = (2, i)
                        vx = corr[neuron_pos.index(npx), neuron_pos.index(npz)]
                        vy = corr[neuron_pos.index(npy), neuron_pos.index(npz)]
                        if np.abs(vx) > th and np.abs(vy) > th:
                            pp[4][npz[1], npx[1]] = 1
                            pp[4][npz[1], npy[1]] = 1
                if npx[0] == 2:
                    for i in range(n):
                        npz = (1, i)
                        vx = corr[neuron_pos.index(npx), neuron_pos.index(npz)]
                        vy = corr[neuron_pos.index(npy), neuron_pos.index(npz)]
                        if np.abs(vx) > th and np.abs(vy) > th:
                            pp[4][npx[1], npz[1]] = 1
                            pp[4][npy[1], npz[1]] = 1

        pp = np.concatenate([x.reshape(-1) for x in pp], axis=0)
        self.pcs = np.where(pp > 0)[0]
        data_dict = {'pcs': self.pcs, 'hyperparam': self.hyperparam}
        np.savez(file, **data_dict)
        return data_dict

    def get_pcs(self, show_iters=(0, 1, 5, 10, 100, 200000)):
        file = self.save_folder + "pcs.npz"
        if os.path.exists(file):
            return np.load(file)

        ss = show_iters
        pcs_dict = {}
        for si, s in enumerate(ss):
            if self.pcs.size == 0:
                self.get_critical_connections(cth=0.5, s=s)
            pcs_dict[f"iter{s}"] = self.pcs
        pcs_dict["hyperparam"] = self.hyperparam
        np.savez(file, **pcs_dict)
        return pcs_dict

    def get_landscape(self, cth=0.5, s=0, p_range=1, n_points=100):
        file = self.save_folder + f"landscape_iter{s}.npz"
        if os.path.exists(file):
            return np.load(file)

        self.update_model(s=s)
        pcs = self.get_critical_connections(cth=cth, s=self.max_iter)['pcs']
        for ii in range(2):
            idx = 200
            while idx in pcs:
                idx = np.random.randint(0, self.data.shape[0])
            self.non_critical_connections[ii] = idx
        self.critical_connections = [pcs[0], pcs[5]]
        ind_pairs = [tuple(self.critical_connections),
                     (self.critical_connections[0], self.non_critical_connections[0]),
                     tuple(self.non_critical_connections)]
        # ind_pairs = [(pcs[0], pcs[5]), (pcs[0], 200), (201, 200)]
        names = ["cc", "cn", "nn"]
        label_pairs = [("Critical", "Critical"), ("Critical", "Non-critical"), ("Non-critical", "Non-critical")]
        data_dict = {}
        n_data = self.data.shape[0]
        model = self.model
        x, y = self.data, self.target
        criterion = self.criterion
        for ii in range(3):
            labels = label_pairs[ii]
            params = [p for p in model.parameters() if p.requires_grad]
            flat_params = flatten_params(model.parameters())
            ind1, ind2 = ind_pairs[ii]
            ori_p1, ori_p2 = flat_params[ind1], flat_params[ind2]
            p_coord1 = torch.linspace(ori_p1 - p_range, ori_p1 + p_range, n_points)
            p_coord2 = torch.linspace(ori_p2 - p_range, ori_p2 + p_range, n_points)
            landscape = np.zeros((n_points, n_points, n_data))
            for i, pi in enumerate(p_coord1):
                for j, pj in enumerate(p_coord2):
                    flat_params[ind1] = float(pi)
                    flat_params[ind2] = float(pj)
                    unflattened_params = unflatten_params(flat_params, params)
                    y_hat = x.reshape(x.shape[0], -1) @ unflattened_params[0].T + unflattened_params[1]
                    y_hat = F.relu(y_hat)
                    y_hat = y_hat @ unflattened_params[2].T + unflattened_params[3]
                    y_hat = F.relu(y_hat)
                    y_hat = y_hat @ unflattened_params[4].T + unflattened_params[5]
                    landscape[i, j] = criterion(y_hat, y).detach().cpu().numpy()

            # effective landscapes
            ll = np.zeros_like(landscape[:, :, 0])
            for i in range(landscape.shape[0]):
                for j in range(landscape.shape[1]):
                    idx = np.random.randint(0, landscape.shape[2], size=(64,))
                    ll[i, j] = np.mean(landscape[i, j, idx])

            X, Y = np.meshgrid(p_coord1, p_coord2)
            data_dict[f"{names[ii]}_X"] = X
            data_dict[f"{names[ii]}_Y"] = Y
            data_dict[f"{names[ii]}_rand_L"] = ll
            data_dict[f"{names[ii]}_L"] = landscape
        data_dict["hyperparam"] = self.hyperparam
        np.savez(file, **data_dict)
        return data_dict

    def get_per_data_loss(self):
        file = self.save_folder + "per_data_loss.npz"
        if os.path.exists(file):
            return np.load(file)

        test_point_indices = self.test_point_indices
        loss_dict = {}
        x, y = self.data, self.target
        for s in test_point_indices:
            self.update_model(s=s)
            flat_params = flatten_params(self.model.parameters())
            unflattened_params = unflatten_params(flat_params, self.params)
            y_hat = x.reshape(x.shape[0], -1) @ unflattened_params[0].T + unflattened_params[1]
            y_hat = F.relu(y_hat)
            y_hat = y_hat @ unflattened_params[2].T + unflattened_params[3]
            y_hat = F.relu(y_hat)
            y_hat = y_hat @ unflattened_params[4].T + unflattened_params[5]
            losses = self.criterion(y_hat, y).detach().cpu().numpy()
            loss_dict[f"iter{s}"] = losses
        np.savez(file, **loss_dict)
        return loss_dict

    def get_conc_ineq(self, b=np.linspace(0, 3, 100)):
        file = self.save_folder + "prob_conc.npz"
        if os.path.exists(file):
            return np.load(file)
        test_point_indices = self.test_point_indices
        n_data = self.data.shape[0]
        prs = {'b': b}
        loss_dict = self.get_per_data_loss()
        for s in test_point_indices:
            losses = loss_dict[f"iter{s}"]
            prs[f"iter{s}"] = []
            for bi in b:
                pr = np.sum(losses >= bi) / n_data
                prs[f"iter{s}"].append(pr)
        prs["hyperparam"] = self.hyperparam
        np.savez(file, **prs)
        return prs

    def get_irreversibility(self):
        file = self.save_folder + "irreversibility.npz"
        if os.path.exists(file):
            return np.load(file)

        path = self.path_data
        irs = []
        for i in range(path.shape[1]):
            t = np.array(self.test_point_indices)
            x = path[:, i]
            dt = np.median(np.diff(np.sort(t)))  # Or choose your desired time step
            t_reg = np.arange(t.min(), t.max(), dt)
            interp_func = scipy.interpolate.interp1d(t, x, kind='linear', fill_value='extrapolate')
            x_reg = interp_func(t_reg)
            # forward_autocorr = correlate(x_reg - x_reg.mean(), x_reg - x_reg.mean(), mode='full')
            forward_autocorr = scipy.signal.correlate(x_reg, x_reg, mode='full')
            # autocorr = autocorr / np.max(autocorr)
            lags = np.arange(-t.max() + dt, t.max(), dt)
            idx = lags > 0
            x_reg_back = x_reg[::-1]
            # backward_autocorr = correlate(x_reg - x_reg.mean(), x_reg_back - x_reg_back.mean(), mode='full')
            backward_autocorr = scipy.signal.correlate(x_reg, x_reg_back, mode='full')
            ir = np.mean((forward_autocorr - backward_autocorr) ** 2)
            irs.append(np.sum(ir))

        np.savez(file, data=irs)
        return irs

    def get_entropies(self):
        file = self.save_folder + "entropies.npz"
        if os.path.exists(file):
            return np.load(file)

        data = self.get_per_data_loss()
        data_dict = {}
        ss = self.test_point_indices
        for si, s in enumerate(ss):
            lk = data[f"iter{s}"]
            if np.isnan(np.mean(lk)):
                entropy = np.ones((100,)) * 1e-10
                temp = np.zeros((100,))
                Bs = np.zeros((100,))
            else:
                Bs = np.linspace(np.mean(lk), np.max(lk), 100)
                entropy = []
                temp = []
                temp_prior = 0
                for b in Bs:
                    t = get_temperature(lk, b, prior_guess=temp_prior)
                    temp_prior = t
                    e = quasi_entropy(lk, t)
                    entropy.append(e)
                    temp.append(t)
            data_dict[f"step{s}_entropy"] = entropy
            data_dict[f"step{s}_temp"] = temp
            data_dict[f"step{s}_Bs"] = Bs
        data_dict['hyperparam'] = self.hyperparam
        np.savez(file, **data_dict)
        return data_dict

    def get_dS_energy(self, bth=5e-3):
        file = self.save_folder + "dS_energy.npz"
        if os.path.exists(file):
            return np.load(file)

        edata = self.get_entropies()
        ldata = self.get_per_data_loss()
        N = self.data.shape[0]
        entropy = []
        energy = []
        ss = self.test_point_indices
        for s in ss:
            lk = ldata[f"iter{s}"]
            if np.isnan(lk) or np.isinf(lk):
                l_bar = 100  # assign a very large value
                e = 1e-10  # assign mimum value
            else:
                l_bar = np.mean(lk)
                e = edata[f"step{s}_entropy"]
                dS = np.log(N) - e
                b = edata[f"step{s}_Bs"]
                spline = UnivariateSpline(b, dS, k=5, s=0)
                e = np.abs(spline(bth))
            entropy.append(e)
            energy.append(l_bar)
        entropy = np.array(entropy)
        energy = np.array(energy)
        lagrangian = np.log(energy) - np.log(entropy)
        data_dict = {
            'dS': entropy,
            'energy': energy,
            'lagrangian': lagrangian,
            'hyperparam': self.hyperparam,
        }
        np.savez(file, **data_dict)
        return data_dict

    def get_corr_mat(self, s=0):
        file = self.save_folder + f"corr_mat_iter{s}.npz"
        if os.path.exists(file):
            return np.load(file)

        spks = self.get_spks(s=s)['spks']
        corr = np_pearson_corr(spks, spks)
        data_dict = {
            'corr': corr,
            'hyperparam': self.hyperparam,
        }
        np.savez(file, **data_dict)
        return data_dict

    def get_all_corr_mat(self):
        file = self.save_folder + f"corr_mat_all.npz"
        if os.path.exists(file):
            return np.load(file)

        data_dict = {}
        for s in self.test_point_indices:
            spks = self.get_spks(s=s)['spks']
            corr = np_pearson_corr(spks, spks)
            data_dict[f"corr_mat{s}"] = corr
        data_dict["hyperparam"] = self.hyperparam
        np.savez(file, **data_dict)
        return data_dict

    def cluster_id_to_weight_dict(self, cluster_id, verbose=False):
        data_dict = {}
        for ti in range(self.num_tasks):
            data_dict[f"task{ti}_edges"] = []
            data_dict[f"task{ti}_neurons"] = []
        # neurons
        for indices in list(cluster_id):
            task = []
            for ni in indices:
                li, pi = self.neuron_pos[ni]
                if li == 2:
                    task.append(pi)  # for regression learning, it is the same, since layer_loc is the single task
            else:
                if len(task) > 1:
                    if verbose:
                        print(f"Multi-task detected {task}, indices {indices}")
                for ti in task:
                    for ni in indices:
                        li, pi = self.neuron_pos[ni]
                        if li == 2:
                            if pi == ti:
                                data_dict[f"task{ti}_neurons"].append(ni)
                        else:
                            data_dict[f"task{ti}_neurons"].append(ni)
        # connections
        for ti in range(self.num_tasks):
            nodes = data_dict[f"task{ti}_neurons"]
            for ni in nodes:
                for nj in nodes:
                    li, pi = self.neuron_pos[ni]
                    lj, pj = self.neuron_pos[nj]
                    if ni == nj:
                        continue
                    if lj == li + 1 and (li, pi, lj, pj) not in data_dict[f"task{ti}_edges"]:
                        data_dict[f"task{ti}_edges"].append((li, pi, lj, pj))
        return data_dict

    def neurons_to_critical_param(self, neurons):
        params = [torch.zeros_like(p) for p in self.params]
        # bias of last layer is not changed
        params[5] = self.params[5]
        for n in neurons:
            li, pi = self.neuron_pos[n]
            if li == 0:
                params[0][pi] = self.params[0][pi]
                params[1][pi] = self.params[1][pi]
            if li == 1:
                params[2][pi] = self.params[2][pi]
                params[3][pi] = self.params[3][pi]
                # receive signal from only critical neurons
                params[4][:, pi] = self.params[4][:, pi]
        return params

    def edges_to_critical_param(self, edges):
        params = [torch.zeros_like(p) for p in self.params]
        # bias of last layer is not changed
        params[5] = self.params[5]
        for cc in edges:
            li, pi, lj, pj = cc  # layer, position
            if li == 0:
                params[0][pi] = self.params[0][pi]
                params[1][pi] = self.params[1][pi]
                params[2][pj, pi] = self.params[2][pj, pi]  # next layer, this layer, since y_hat = x @ W.T
                params[3][pj] = self.params[3][pj]
            if li == 1:
                # the prediction should include all tasks
                # otherwise all predictions are perfect
                # receive signal from only critical neurons
                params[4][:, pi] = self.params[4][:, pi]
        return params

    def weight_dict_to_graph(self, weight_dict):
        init, lr, bs, neurons = self.hyperparam
        pos = np.linspace(0, neurons, self.num_tasks)
        corr = self.get_corr_mat(s=self.max_iter)['corr']
        graphs = [nx.Graph() for _ in range(self.num_tasks)]
        sorts = []
        for ti in range(self.num_tasks):
            graph = graphs[ti]
            sort = []
            connections = weight_dict[f"task{ti}_edges"]
            for cc in connections:
                li, pi, lj, pj = cc  # layer, position
                ni = self.neuron_pos.index((li, pi))  # row index of correlation matrix
                nj = self.neuron_pos.index((lj, pj))
                sort.append(ni)
                graph.add_node(
                    f"{li}_{pi}_{ni}",
                    value=ni,
                    coordinates=(li, pi),
                    color='red',
                )
                sort.append(nj)
                if lj == 2:
                    graph.add_node(
                        f"{lj}_{pj}_{nj}",
                        value=nj,
                        coordinates=(lj, pos[pj]),
                        color='red',
                    )
                else:
                    graph.add_node(
                        f"{lj}_{pj}_{nj}",
                        value=nj,
                        coordinates=(lj, pj),
                        color='red',
                    )
                color = color_list[4] if corr[ni, nj] > 0 else color_list[5]
                graph.add_edge(
                    f"{li}_{pi}_{ni}",
                    f"{lj}_{pj}_{nj}",
                    value=corr[ni, nj],
                    color=color,
                )
            sorts.append(np.unique(sort))
        sorts = [s for sort in sorts for s in sort]
        for ni in range(corr.shape[0]):
            if ni not in sorts:
                sorts.append(ni)
        return graphs, sorts

    def get_ccg_umap_cluster(self, s=None, neighbors=2):
        if not s:
            s = self.max_iter
        file = self.save_folder + "ccg_umap_cluster.npz"
        if os.path.exists(file):
            return np.load(file, allow_pickle=True)

        data_dict = {}
        corr = self.get_corr_mat(s=s)['corr']
        reducer = umap.UMAP(n_neighbors=neighbors, min_dist=0, n_components=2, metric='precomputed')
        corr1 = np.exp(5 * (1 - corr))  # consider only positively correlated neurons
        embedding = reducer.fit_transform(corr1)

        init, lr, bs, neurons = self.hyperparam
        max_n = min(50, 2*neurons)
        min_n = 2
        range_n_clusters = np.arange(min_n, max_n)
        s_scores = []
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            cluster_labels = kmeans.fit_predict(embedding)
            silhouette_avg = silhouette_score(embedding, cluster_labels)
            s_scores.append(silhouette_avg)
        if len(s_scores) > 0:
            best_n_clusters = range_n_clusters[np.argsort(s_scores)[-1]]
        else:
            best_n_clusters = neurons

        print(f"best number of clusters for umap embedding: {best_n_clusters}")
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init="auto").fit(embedding)
        u_nodes = kmeans.labels_
        c_nodes = kmeans.cluster_centers_

        if len(c_nodes) > 1:
            cc = np.zeros((len(c_nodes), len(c_nodes)))
            for i in range(len(c_nodes)):
                for j in range(len(c_nodes)):
                    cc[i, j] = (c_nodes[i] - c_nodes[j]) @ (c_nodes[i] - c_nodes[j]).T
            # cc = 1 / (cc+1e-5)
            # cc = c_nodes @ c_nodes.T
            cc = np.exp(-cc / len(c_nodes))
            cc, inds = traveling_salesman(cc, verbose=False, locality=0.75, n_skip=None)[:2]
            c_nodes = c_nodes[inds]
        else:
            c_nodes = c_nodes
        cluster_id = []
        for ci in np.unique(u_nodes):
            indices = np.where(u_nodes == ci)[0]
            cluster_id.append(indices)
        data_dict["embedding"] = embedding
        data_dict["u_nodes"] = u_nodes
        data_dict["cluster_id"] = np.array(cluster_id, dtype=object)
        data_dict["cluster_center"] = c_nodes
        data_dict["hyperparam"] = self.hyperparam
        np.savez(file, **data_dict)
        return data_dict

    def get_ccg_clique_cluster(self, s=None, clique_neighbors=8):
        if not s:
            s = self.max_iter
        file = self.save_folder + "ccg_clique_cluster.npz"
        if os.path.exists(file):
            return np.load(file, allow_pickle=True)

        data_dict = {}
        corr = self.get_corr_mat(s=s)['corr']
        G = nx.Graph()
        n = corr.shape[0]
        G.add_nodes_from(range(n))
        inds = fast_knn_indices_from_precomputed(1 - corr, n_neighbors=clique_neighbors)
        for ind in inds:
            for i in ind:
                for j in ind:
                    G.add_edge(i, j, weight=corr[i, j])

        cliques = nx.find_cliques(G)
        cliques = list(cliques)
        data_dict["cluster_id"] = np.array(cliques, dtype=object)
        data_dict["hyperparam"] = self.hyperparam
        np.savez(file, **data_dict)
        return data_dict

    def get_ccg_critical_neurons(self, s=None, loops=2):
        file = self.save_folder + "ccg_critical_neurons.npz"
        if os.path.exists(file):
            return np.load(file)

        data_dict = {}
        if not s:
            s = self.max_iter
        self.update_model(s=s)
        umap_id = self.get_ccg_umap_cluster(s=s)["cluster_id"]
        clique_id = self.get_ccg_clique_cluster(s=s)["cluster_id"]
        umap_dict = self.cluster_id_to_weight_dict(umap_id)
        clique_dict = self.cluster_id_to_weight_dict(clique_id)

        for ti in range(self.num_tasks):
            data_dict[f"task{ti}_edges"] = []
            idx = torch.where(self.target == ti)
            x, y = self.data[idx], self.target[idx]
            # from shallow to deep
            init_neurons = list(umap_dict[f"task{ti}_neurons"])
            target_neurons = list(clique_dict[f"task{ti}_neurons"])
            target_neurons = sorted(target_neurons)

            init_params = self.neurons_to_critical_param(init_neurons)
            # measure initial performance
            loss_react = []
            acc_react = []
            with torch.no_grad():
                y_hat = x.reshape(x.shape[0], -1)
                y_hat = y_hat @ init_params[0].T + init_params[1]
                y_hat = F.relu(y_hat)
                y_hat = y_hat @ init_params[2].T + init_params[3]
                y_hat = F.relu(y_hat)
                y_hat = y_hat @ init_params[4].T + init_params[5]
                loss = self.criterion(y_hat, y).mean()  # for using none reduction
                _, predicted = torch.max(y_hat.data, 1)
                correct = (predicted == y).sum().item() / x.shape[0]
                loss_react.append(loss.item())
                acc_react.append(correct)
            print(f"critical neurons: task {ti}, init neurons {len(init_neurons)} base loss {loss.item()}, base acc {correct}")

            for count_loop in range(loops):
                for nt in target_neurons:
                    if nt in init_neurons:
                        continue
                    init_neurons.append(nt)
                    params = self.neurons_to_critical_param(init_neurons)
                    with torch.no_grad():
                        y_hat = x.reshape(x.shape[0], -1)
                        y_hat = y_hat @ params[0].T + params[1]
                        y_hat = F.relu(y_hat)
                        y_hat = y_hat @ params[2].T + params[3]
                        y_hat = F.relu(y_hat)
                        y_hat = y_hat @ params[4].T + params[5]
                        loss = self.criterion(y_hat, y).mean()  # for using none reduction
                        _, predicted = torch.max(y_hat.data, 1)
                        correct = (predicted == y).sum().item() / x.shape[0]
                        # accuracy is a coarse criteria, use loss to maximize performance
                        if correct > acc_react[-1]:
                            loss_react.append(loss.item())
                            acc_react.append(correct)
                        else:
                            init_neurons.pop()
                print(f"critical neurons: task {ti}, loop {count_loop}, optimal neurons {len(init_neurons)}, optimal loss {loss.item()}, optimal acc {correct}")
            data_dict[f"task{ti}_neurons"] = init_neurons
            for ni in init_neurons:
                for nj in init_neurons:
                    li, pi = self.neuron_pos[ni]
                    lj, pj = self.neuron_pos[nj]
                    if ni == nj:
                        continue
                    if lj == li + 1 and (li, pi, lj, pj) not in data_dict[f"task{ti}_edges"]:
                        data_dict[f"task{ti}_edges"].append((li, pi, lj, pj))
        np.savez(file, **data_dict)
        return data_dict

    def get_ccg_critical_edges(self, s=None):
        file = self.save_folder + "ccg_critical_edges.npz"
        if os.path.exists(file):
            return np.load(file)

        data_dict = {}
        if not s:
            s = self.max_iter
        self.update_model(s=s)
        init, lr, bs, neurons = self.hyperparam
        data = self.get_ccg_critical_neurons(s=s)
        for ti in range(self.num_tasks):
            data_dict[f"task{ti}_neurons"] = []
            idx = torch.where(self.target == ti)
            x, y = self.data[idx], self.target[idx]
            init_edges = data[f"task{ti}_edges"]
            init_edges = [tuple(edge) for edge in init_edges]
            init_edges = sorted(init_edges, key=lambda s: s[2])[::-1]  # from deep to shallow
            init_params = self.params
            loss_react = []
            acc_react = []
            with torch.no_grad():
                y_hat = x.reshape(x.shape[0], -1)
                y_hat = y_hat @ init_params[0].T + init_params[1]
                y_hat = F.relu(y_hat)
                y_hat = y_hat @ init_params[2].T + init_params[3]
                y_hat = F.relu(y_hat)
                y_hat = y_hat @ init_params[4].T + init_params[5]
                loss = self.criterion(y_hat, y).mean()  # for using none reduction
                _, predicted = torch.max(y_hat.data, 1)
                correct = (predicted == y).sum().item() / x.shape[0]
                loss_react.append(loss.item())
                acc_react.append(correct)
                print(f"critical edges: task {ti}, init edges {len(init_edges)}, base loss {loss.item()}, base acc {correct}")
            for _ in range(len(init_edges)):
                pop_ind = init_edges.pop(0)  # remove one edge to perturb NN
                if pop_ind[2] == 2:
                    init_edges.append(pop_ind)  # last layer does not have pruning
                    continue
                # ensure consistency of computational graph
                edges = []
                for edge in init_edges:
                    li, pi, lj, pj = edge
                    if li > 0:
                        linked = False
                        for p in range(neurons):
                            if (li - 1, p, li, pi) in init_edges:
                                linked = True
                    if lj < 2:
                        linked = False
                        for p in range(neurons):
                            if (lj, pj, lj + 1, p) in init_edges:
                                linked = True
                    if linked:
                        edges.append(edge)
                params = self.edges_to_critical_param(edges)
                with torch.no_grad():
                    y_hat = x.reshape(x.shape[0], -1)
                    y_hat = y_hat @ params[0].T + params[1]
                    y_hat = F.relu(y_hat)
                    y_hat = y_hat @ params[2].T + params[3]
                    y_hat = F.relu(y_hat)
                    y_hat = y_hat @ params[4].T + params[5]
                    loss = self.criterion(y_hat, y).mean()  # for using none reduction
                    _, predicted = torch.max(y_hat.data, 1)
                    correct = (predicted == y).sum().item() / x.shape[0]
                if correct >= acc_react[-1]:
                    init_edges = edges
                else:
                    init_edges.append(pop_ind)
            print(f"critical edges: task {ti}, edges after pruning: {len(init_edges)}, optimal loss {loss.item()}, optimal acc {correct}")
            data_dict[f"task{ti}_edges"] = init_edges
            for ee in init_edges:
                li, pi, lj, pj = ee
                ni = self.neuron_pos.index((li, pi))
                data_dict[f"task{ti}_neurons"].append(ni)
                if lj == 2:
                    nj = self.neuron_pos.index((lj, pj))
                    data_dict[f"task{ti}_neurons"].append(nj)

        np.savez(file, **data_dict)
        return data_dict

    def get_ccg_performance(
            self,
            weight_dict,
            s=None,
            label="prune",  # or umap, or clique
            data_loader="train",  # or test
            ):
        file = self.save_folder + f"ccg_loss_acc_{label}_{data_loader}.npz"
        if os.path.exists(file):
            return np.load(file)

        if not s:
            s = self.max_iter
        self.update_model(s=s)
        data_dict = {}
        reserve_neurons = ("all", "critical", "non-critical")

        for ti in range(self.num_tasks):
            with torch.no_grad():
                if data_loader == "train":
                    idx = torch.where(self.target == ti)
                    x, y = self.data[idx], self.target[idx]
                else:
                    idx = torch.where(self.test_target == ti)
                    x, y = self.test_data[idx], self.test_target[idx]
                for reserve in reserve_neurons:
                    if reserve == "all":
                        params = self.params
                    if reserve == "critical":
                        if label in ["umap", "clique", "critical neurons"]:
                            params = self.neurons_to_critical_param(weight_dict[f"task{ti}_neurons"])
                        else:
                            params = self.edges_to_critical_param(weight_dict[f"task{ti}_edges"])
                    if reserve == "non-critical":
                        if label in ["umap", "clique", "critical neurons"]:
                            params = self.neurons_to_critical_param(weight_dict[f"task{ti}_neurons"])
                        else:
                            params = self.edges_to_critical_param(weight_dict[f"task{ti}_edges"])
                        masks = [p != 0 for p in params]
                        params = []
                        ori_params = copy.deepcopy(self.params)
                        for p, mask in zip(ori_params, masks):
                            p[mask] = 0
                            params.append(p)
                    y_hat = x.reshape(x.shape[0], -1)
                    y_hat = y_hat @ params[0].T + params[1]
                    y_hat = F.relu(y_hat)
                    y_hat = y_hat @ params[2].T + params[3]
                    y_hat = F.relu(y_hat)
                    y_hat = y_hat @ params[4].T + params[5]

                    loss = self.criterion(y_hat, y)  # record per-data loss
                    _, predicted = torch.max(y_hat.data, 1)
                    correct = (predicted == y).sum().item() / x.shape[0]
                    data_dict[f"task{ti}_reserve_{reserve}_loss"] = loss.detach().cpu().numpy()
                    data_dict[f"task{ti}_reserve_{reserve}_acc"] = correct
        np.savez(file, **data_dict)
        return data_dict

    def weight_dict_to_param(self, weight_dict, content="neurons"):
        params = [torch.zeros_like(p) for p in self.params]
        if content == "neurons":
            neurons = []
            for ti in range(self.num_tasks):
                ns = list(weight_dict[f"task{ti}_neurons"])
                neurons += ns
            # bias of last layer is not changed
            params[5] = self.params[5]
            for n in neurons:
                li, pi = self.neuron_pos[n]
                if li == 0:
                    params[0][pi] = self.params[0][pi]
                    params[1][pi] = self.params[1][pi]
                if li == 1:
                    params[2][pi] = self.params[2][pi]
                    params[3][pi] = self.params[3][pi]
                    # receive signal from only critical neurons
                    params[4][:, pi] = self.params[4][:, pi]
        if content == "edges":
            edges = []
            for ti in range(self.num_tasks):
                es = list(weight_dict[f"task{ti}_edges"])
                edges += es
            params = [torch.zeros_like(p) for p in self.params]
            # bias of last layer is not changed
            params[5] = self.params[5]
            for cc in edges:
                li, pi, lj, pj = cc  # layer, position
                if li == 0:
                    params[0][pi] = self.params[0][pi]
                    params[1][pi] = self.params[1][pi]
                    params[2][pj, pi] = self.params[2][pj, pi]  # next layer, this layer, since y_hat = x @ W.T
                    params[3][pj] = self.params[3][pj]
                if li == 1:
                    # the prediction should include all tasks
                    # otherwise all predictions are perfect
                    # receive signal from only critical neurons
                    params[4][:, pi] = self.params[4][:, pi]
        return params

    # TODO: for general input, we gather the prediction of each specific ccg,
    #  and choose the one with the smallest prediction entropy
    def get_ccg_performance_all_task(
            self,
            weight_dict,
            s=None,
            label="prune",  # or umap, or clique
            data_loader="train",  # or test
            ):
        file = self.save_folder + f"ccg_loss_acc_all_task_{label}_{data_loader}.npz"
        if os.path.exists(file):
            return np.load(file)

        if not s:
            s = self.max_iter
        self.update_model(s=s)
        data_dict = {}
        reserve_neurons = ("all", "critical", "non-critical")

        with torch.no_grad():
            if data_loader == "train":
                x, y = self.data, self.target
            else:
                x, y = self.test_data, self.test_target
            for reserve in reserve_neurons:
                if reserve == "all":
                    params = self.params
                if reserve == "critical":
                    if label in ["umap", "clique", "critical neurons"]:
                        # gather all neurons
                        params = self.weight_dict_to_param(weight_dict, "neurons")
                    else:
                        # gather all edges
                        params = self.weight_dict_to_param(weight_dict, "edges")
                if reserve == "non-critical":
                    if label in ["umap", "clique", "critical neurons"]:
                        params = self.weight_dict_to_param(weight_dict, "neurons")
                    else:
                        params = self.weight_dict_to_param(weight_dict, "edges")
                    masks = [p != 0 for p in params]
                    params = []
                    ori_params = copy.deepcopy(self.params)
                    for p, mask in zip(ori_params, masks):
                        p[mask] = 0
                        params.append(p)
                y_hat = x.reshape(x.shape[0], -1)
                y_hat = y_hat @ params[0].T + params[1]
                y_hat = F.relu(y_hat)
                y_hat = y_hat @ params[2].T + params[3]
                y_hat = F.relu(y_hat)
                y_hat = y_hat @ params[4].T + params[5]

                loss = self.criterion(y_hat, y)  # record per-data loss
                _, predicted = torch.max(y_hat.data, 1)
                correct = (predicted == y).sum().item() / x.shape[0]
                data_dict[f"reserve_{reserve}_loss"] = loss.detach().cpu().numpy()
                data_dict[f"reserve_{reserve}_acc"] = correct
        np.savez(file, **data_dict)
        return data_dict

    def get_loss_trajectory(self):
        file = self.save_folder + "loss_trajectory.npz"
        if os.path.exists(file):
            return np.load(file)

        data = self.get_per_data_loss()
        ss = self.test_point_indices
        N = self.data.shape[0]
        trajectories = np.zeros((N, len(ss)))
        for si, s in enumerate(ss):
            lk = data[f"iter{s}"]
            trajectories[:, si] = lk
        data_dict = {'trajectories': trajectories}
        np.savez(file, **data_dict)
        return data_dict

    @staticmethod
    def get_marginal_velocity(traj, ts):
        dx = np.diff(traj, axis=1)
        dt = np.diff(ts)
        return dx / dt

    def get_marginal_flux(self, traj, bins=100):
        N = traj.shape[0]
        ss = self.test_point_indices
        min_v, max_v = np.min(traj), np.max(traj)
        edges = np.linspace(min_v, max_v, bins + 1)
        x = traj[:, :-1]
        v = self.get_marginal_velocity(traj, ss)
        bin_idx = np.digitize(x, edges, right=True) - 1
        valid = (bin_idx >= 0) & (bin_idx < bins)  # Mask out-of-range values to -1
        bin_idx = np.where(valid, bin_idx, -1)
        flux = np.zeros((bins, len(ss) - 1), dtype=v.dtype)
        lin_idx = bin_idx * (len(ss) - 1) + np.broadcast_to(np.arange(len(ss) - 1), bin_idx.shape)
        flux_flat = flux.ravel()
        np.add.at(flux_flat, lin_idx[valid], v[valid])
        flux = flux_flat.reshape(bins, len(ss) - 1)
        flux /= N
        return flux

    def get_loss_flux(self, bins=100, overwrite=False):
        file = self.save_folder + "flux_loss.npz"
        if os.path.exists(file) and not overwrite:
            return np.load(file)

        data_dict = {}
        trajectories = self.get_loss_trajectory()['trajectories']
        flux = self.get_marginal_flux(trajectories, bins=bins)
        # ss = self.test_point_indices
        # flux = np.zeros((bins, len(ss) - 1))  # J(L, t)
        # trajectories = self.get_loss_trajectory()['trajectories']
        # velocities = self.get_marginal_velocity(trajectories, ss)
        # min_v, max_v = np.min(trajectories), np.max(trajectories)
        # edges = np.linspace(min_v, max_v, bins + 1)
        # N = self.data.shape[0]
        # for ti, t in enumerate(ss[:-1]):
        #     x = trajectories[:, ti]
        #     for i in range(bins):
        #         idx = np.where((x > edges[i]) & (x <= edges[i + 1]))
        #         flux[i, ti] = np.sum(velocities[idx, ti]) / N
        data_dict["flux"] = flux
        np.savez(file, **data_dict)
        return data_dict

    def get_weight_flux(self, bins=100, overwrite=False):
        file = self.save_folder + "flux_weight.npz"
        if os.path.exists(file) and not overwrite:
            return np.load(file)

        data_dict = {}
        trajectories = self.path_data.T
        flux = self.get_marginal_flux(trajectories, bins=bins)
        # N = trajectories.shape[0]
        # ss = self.test_point_indices
        # flux = np.zeros((bins, len(ss) - 1))  # J(L, t)
        # velocities = self.get_marginal_velocity(trajectories, ss)
        # min_v, max_v = np.min(trajectories), np.max(trajectories)
        # edges = np.linspace(min_v, max_v, bins + 1)
        # data_dict = {}
        # for ti, t in enumerate(ss[:-1]):
        #     x = trajectories[:, ti]
        #     for i in range(bins):
        #         idx = np.where((x > edges[i]) & (x <= edges[i + 1]))
        #         flux[i, ti] = np.sum(velocities[idx, ti]) / N
        data_dict["flux"] = flux
        np.savez(file, **data_dict)
        return data_dict

    def weight_from_neurons(self, Ni, Nj):
        li, pi = self.neuron_pos[Ni]  # layer, position
        lj, pj = self.neuron_pos[Nj]
        wij = torch.tensor([0])
        if li == 0 and lj == 1:
            wij = self.params[2][pj, pi]  # next layer, this layer, since y_hat = x @ W.T
        if lj == 0 and li == 1:
            wij = self.params[2][pi, pj]
        if li == 1 and lj == 2:
            wij = self.params[4][pj, pi]
        if lj == 1 and li == 2:
            wij = self.params[4][pi, pj]
        return wij.detach().cpu().numpy()

    def weight_evolution_sim_single(self, p_Hebb=0.37, n_traj=100):
        file = self.save_folder + f"hebb_sim_empirical_corr_dist_data_p{p_Hebb}.npz"
        if os.path.exists(file):
            return np.load(file)

        init, lr, bs, neurons = self.hyperparam
        corr = self.get_all_corr_mat()
        ts = np.zeros((len(corr),))
        Cs = np.zeros((len(corr), neurons, neurons))
        for i, k in enumerate(corr):
            if "corr" in k:
                t = k.strip("corr_mat")
                ts[i] = t
                Cs[i] = corr[k]

        realmin = 1e-30
        # architecture
        N = neurons
        E = int(N * (N - 1) / 2)
        I, J = np.triu_indices(N, k=1)
        print(f"Simulate correlation-dependent model with p_Hebb: {p_Hebb}")
        # hyperparameter
        lr = 1  # use discrete value to ensure numerical stability
        p_update = 0.01 / lr
        n_update = int(E * p_update)
        total_num_updates = self.max_iter

        W = np.zeros((N, N))
        corr_func = scipy.interpolate.interp1d(ts, Cs, axis=0)
        C = corr["corr_mat0"]
        prob = np.abs(C[I, J]) + realmin
        prob /= prob.sum()
        inds_inc = np.random.choice(E, int(0.1 / lr * E), replace=True, p=prob)
        row, col = I[inds_inc], J[inds_inc]
        W = W + lr * csc_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N)).toarray()

        # record data
        data_dict = {}
        data_dict["p"] = p_Hebb
        data_dict["W0"] = W
        data_dict["C0"] = C

        for t in tqdm(range(1, total_num_updates + 1)):
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
            prob = np.abs(C[I, J]) + realmin
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
        np.savez(self.save_folder + f"hebb_sim_data_empirical_corr_p{p_Hebb}.npz", **data_dict)

        W = data_dict[f"W{self.max_iter}"]
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
                prob = np.abs(W[I, J]) + realmin
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
        np.savez(file, Ws=Ws)
        return Ws

    def weight_evolution_sim(self, ps=np.linspace(0.1, 0.9, 50)):
        for p in ps:
            self.weight_evolution_sim_single(p_Hebb=p)

    def get_ccg_coupling_energy_evolve(self):
        file = self.save_folder + "ccg_coupling_energy_evolve.npz"
        if os.path.exists(file):
            return np.load(file)

        data_dict = {}
        for t in range(self.num_tasks):
            es = self.get_ccg_critical_edges()[f"task{t}_edges"]
            E = []
            for s in self.test_point_indices:
                self.update_model(s=s)
                spks = self.get_spks(s=s)['spks']
                num_data = self.get_spks(s=s)['num_data']
                num_data = np.cumsum(num_data)
                num_data = np.concatenate([np.array([0]), num_data], axis=0)
                ss = 0
                for e in es:
                    li, pi, lj, pj = e
                    i = self.neuron_pos.index((li, pi))
                    j = self.neuron_pos.index((lj, pj))
                    wij = self.weight_from_neurons(i, j)
                    si = spks[i]
                    sj = spks[j]
                    si = (si - np.mean(si)) / np.sqrt(np.var(si))
                    sj = (sj - np.mean(sj)) / np.sqrt(np.var(sj))
                    wsi = wij * si * sj
                    ss = ss + wsi
                a, b = num_data[t], num_data[t + 1]
                E.append(-np.mean(ss[a:b]))
            data_dict[f"task{t}"] = E
        np.savez(file, **data_dict)
        return data_dict











