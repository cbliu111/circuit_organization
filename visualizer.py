import os
import matplotlib.pyplot as plt
from utils import *
from scipy.optimize import curve_fit
from rastermap import Rastermap, utils
import networkx as nx
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.signal import correlate
from analyzer import NNAnalyzer


class NNVisualizer(NNAnalyzer):
    """
    compute_hessian_block is used for a 1 hidden layer fully-connected network, other type of network should
    implement the forward computation in this function.
    Assume criterion use reduction type 'none'.
    Only support pytorch dataloader.
    """

    def __init__(self,
                 max_iter=200000,
                 hyperparam=(0, 0.1, 64, 50),
                 test_point_indices=None,
                 device='cpu',
                 data_folder="./autodl-tmp/",
                 save_folder="./autodl-tmp/figures/",
                 num_workers=0,
                 ):
        super().__init__()
        self.device = device
        self.pcs = None  # critical connections
        self.alphas = None  # activity power law exponents
        self.data_folder = data_folder
        self.prefix = save_folder

        self.max_iter = max_iter
        if test_point_indices:
            self.test_point_indices = test_point_indices
        else:
            self.test_point_indices = [i for i in range(0, 50)] + [i for i in range(50, 500, 10)] + [i for i in range(500, max_iter, 500)] + [max_iter]
        self.cth = np.linspace(0.5, 1, 10)  # correlation thresholds
        self.lognorm = LogNorm(vmin=1, vmax=max_iter)
        self.norm01 = Normalize(vmin=0, vmax=1)
        self.iter_cmap = plt.get_cmap('jet')
        self.num_workers = num_workers
        self.init_dataset()
        self.set_hyperparam(*hyperparam)
        self.init_model()

    def rastermap(self, show_iters=(0, 1, 5, 10, 100, 200000)):
        # get neuron indices from last iteration
        spks = self.get_spks(s=self.max_iter)['spks']
        model = Rastermap(n_clusters=100,  # number of clusters to compute
                          n_PCs=128,  # number of PCs to use
                          locality=0.9,  # locality in sorting to find sequences (this is a value from 0-1)
                          time_lag_window=0,  # use future timepoints to compute correlation
                          grid_upsample=10,  # default value, 10 is good for large recordings
                          bin_size=1,
                          symmetric=True,
                          ).fit(spks)
        y = model.embedding  # neurons x 1
        isort = model.isort
        np.savez(self.save_folder + "isort.npz", isort=isort)
        ss = show_iters
        for s in ss:
            data = self.get_spks(s=s)
            spks = data['spks']
            num_data = data['num_data']
            nbin = 1  # number of neurons to bin over
            sn = utils.bin1d(spks[isort], bin_size=nbin, axis=0)  # bin over neuron axis
            # only plot the first 50 images
            start_idx = np.array(num_data).cumsum()[:-1]
            idx = np.concatenate([np.arange(200)] + [np.arange(200) + i for i in start_idx], axis=0)
            # plot_data = np.concatenate([data, sn[:, idx]], axis=0)
            plot_data = sn[:, idx]
            fig, ax = plt.subplots(1, 1, figsize=(3, 2))
            plt.imshow(plot_data, cmap="gray_r", vmin=0, vmax=0.8, aspect="auto")
            # for i in range(10):
            #     plt.text(x=i * 256 + 120, y=-20, s=f"{i}", fontdict={'size': 36})
            ax.set_axis_off()
            plt.savefig(self.save_folder + f"mnist_activity_rastermap_iter{s}.svg",
                        dpi=600, bbox_inches="tight", transparent=True)
            plt.close()
        self.info("Rastermap")

    def correlation_matrix(self, show_iters=(0, 1, 5, 10, 100, 200000)):
        ss = show_iters
        cmap = plt.get_cmap('seismic')
        if not os.path.exists(self.save_folder + "isort.npz"):
            self.rastermap()
        for s in ss:
            spks = self.get_spks(s=s)['spks']
            isort = np.load(self.save_folder + f"isort.npz")["isort"]
            fig, ax = plt.subplots(1, 1, figsize=(3, 2))
            corr = np_pearson_corr(spks[isort], spks[isort])
            cax = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(self.save_folder + f"mnist_activity_corr_iter{s}.svg",
                        dpi=600, bbox_inches="tight", transparent=True)
            plt.close()
        self.info("Correlation matrix")

    def cumulative_explained_variance(self):
        set_figure(4)
        init, lr, bs, neurons = self.hyperparam
        cmap = self.iter_cmap
        fig, ax = plt.subplots()
        for s in self.test_point_indices:
            data = self.get_activity_measures(s=s)
            pc_indices = data['pc_indices']
            cumulative_explained_variance = data['cumulative_explained_variance']
            color = cmap(self.lognorm(s + 1))
            plt.plot(pc_indices, cumulative_explained_variance, color=color, lw=1)
        sm = ScalarMappable(cmap=cmap, norm=self.lognorm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        # cbar.ax.set_position([0.55, 0.1, 0.5, 0.03])
        cbar.set_label(r'$t$')
        cbar.set_ticks([])
        plt.xlabel(r"$n$")
        plt.ylabel(r"$\sigma$")
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + "mnist_activity_cumulative_explained_variance.svg",
                    dpi=600, bbox_inches='tight', transparent=True)
        plt.close()
        self.info("Cumulative explained variance")

    def cumulative_explained_variance_PC(self, pcs=25):
        set_figure(4)
        fig, ax = plt.subplots()
        cmap = self.iter_cmap
        init, lr, bs, neurons = self.hyperparam
        norm = Normalize(vmin=1, vmax=pcs)
        cevs = []
        for s in self.test_point_indices:
            data = self.get_activity_measures(s=s)
            cumulative_explained_variance = data['cumulative_explained_variance']
            cevs.append(cumulative_explained_variance)
        cevs = np.stack(cevs)
        for i in range(pcs):
            color = cmap(norm(i))
            plt.plot(self.test_point_indices, cevs[:, i], lw=1, color=color)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$n$')  # principle direction
        cbar.set_ticks([])
        plt.xscale('log')
        plt.xlabel(r"$t$")  # iteration
        plt.ylabel(r"$\sigma$")  # cumulative explained variance
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(self.save_folder + "mnist_activity_cumulative_explained_variance_PC.svg",
                    dpi=600, bbox_inches='tight', transparent=True)
        plt.close()
        self.info("Cumulative explained variance PC")

    def activity_power_law(self):
        set_figure(4)
        init, lr, bs, neurons = self.hyperparam
        cmap = self.iter_cmap
        fig, ax = plt.subplots()
        for s in self.test_point_indices:
            data = self.get_activity_measures(s=s)
            normalized_eigenvalues = data['normalized_eigenvalues']
            pc_indices = data['pc_indices']
            color = cmap(self.lognorm(s + 1))
            plt.loglog(pc_indices, normalized_eigenvalues, linewidth=1, color=color)
        sm = ScalarMappable(cmap=cmap, norm=self.lognorm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$t$')
        cbar.set_ticks([])
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"$n$")
        plt.ylabel(r"$\lambda$")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(self.save_folder + f"mnist_activity_power_law.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Activity power law")

    def activity_power_law_zoom(self):
        set_figure(4)
        fig, ax = plt.subplots()
        init, lr, bs, neurons = self.hyperparam
        cmap = self.iter_cmap
        self.alphas = []
        for s in self.test_point_indices:
            data = self.get_activity_measures(s=s)
            normalized_eigenvalues = data['normalized_eigenvalues']
            pc_indices = data['pc_indices']
            alpha, ypred = get_powerlaw(normalized_eigenvalues, np.arange(4, 50).astype(int))
            self.alphas.append(alpha)
            color = cmap(self.lognorm(s + 1))
            plt.loglog(pc_indices[:50], normalized_eigenvalues[:50], linewidth=1, color=color)
        sm = ScalarMappable(cmap=cmap, norm=self.lognorm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$\ln t$')  # iteration
        cbar.set_ticks([])
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"$n$")  # PC dimension
        plt.ylabel(r"$\lambda$")  # eigenvalues
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(self.save_folder + "mnist_activity_power_law_zoom.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("mnist_activity_power_law_zoom")

    def activity_sparsity(self):
        set_figure(4)
        fig, ax = plt.subplots()
        if not self.alphas:
            self.activity_power_law_zoom()
        plt.plot(self.test_point_indices, self.alphas, lw=1, color=color_list[0])
        plt.xscale("log")
        plt.xlabel(r"$t$")
        plt.ylabel("Sparsity")
        plt.savefig(self.save_folder + "mnist_activity_power_law_exponent.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Activity sparsity")

    def corr_dist(self):
        set_figure(4)
        fig, ax = plt.subplots()
        cmap = self.iter_cmap
        for si, s in enumerate([0, 100, 200000]):
            spks = self.get_spks(s=s)['spks']
            corr = np_pearson_corr(spks, spks)
            color = cmap(self.lognorm(s + 1))
            # h1 = plt.hist(np.abs(corr[corr > 0.01]).reshape(-1), bins=300, alpha=1,
            #               color=color, density=True, histtype='step', linewidth=1, label="Correlation")
            idx = np.where((corr > 0.1) & (corr < 0.99))
            p, e = hist(np.abs(corr[idx]).reshape(-1), bins=100)
            ax.plot(e, p, lw=1, color=color, ls='none', marker='.', ms=10, alpha=0.5)
        sm = ScalarMappable(cmap=cmap, norm=self.lognorm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$\ln t$')
        cbar.set_ticks([])
        plt.yscale('log')
        plt.xscale('log')
        # plt.legend()
        plt.xlabel(r"$|C_{ij}|$")  # Correlation
        plt.ylabel(r"$p(|C_{ij}|)$")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([0.09, 1.1])
        plt.savefig(self.save_folder + "mnist_activity_corr_dist.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("mnist_activity_corr_dist")

    def cross_corr(self):
        set_figure(3)
        fig, axes = plt.subplots(10, 10, figsize=(7.2, 7))
        init, lr, bs, neurons = self.hyperparam
        cmap = self.iter_cmap
        for s in [0, 100, 200000]:
            data = self.get_spks(s=s)
            sn = data['spks']
            label = data['labels']
            for i1 in range(10):
                for i2 in range(10):
                    ax = axes[i1, i2]
                    if i1 < i2:
                        ax.axis("off")
                        continue
                    idx1 = label == i1
                    idx2 = label == i2
                    n_samples = min(np.array(idx1).sum(), np.array(idx2).sum())
                    sn1 = sn[:, idx1][:, :n_samples]
                    sn2 = sn[:, idx2][:, :n_samples]
                    corr = np_pearson_corr(sn1, sn2)
                    color = cmap(self.lognorm(s + 1))
                    idx = np.where(np.abs(corr) < 0.99)
                    p, e = hist(corr[idx].reshape(-1), bins=100)
                    ax.plot(e, p, lw=1, color=color, alpha=0.5)

                    # h = ax.hist(corr.reshape(-1), bins=300, alpha=0.7,
                    #       color=color, density=True, histtype='step', linewidth=2, label="Correlation")
                    ax.set_yscale('log')
                    # ax.set_xscale('log')

                    # if i1 < 9 and i2 > 0:
                    #     ax.set_xticks([])
                    #     ax.set_yticks([])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_ylim([1e-4, 0.5])
                    if i1 == 9 and i2 != 0:
                        ax.set_yticks([])
                        ax.set_xlabel(f"{i2}")
                    if i2 == 0 and i1 != 9:
                        ax.set_xticks([])
                        ax.set_ylabel(f"{i1}")
                    if i2 == 0 and i1 == 9:
                        ax.set_xlabel(f"{i2}")
                        ax.set_ylabel(f"{i1}")

        sm = ScalarMappable(cmap=cmap, norm=self.lognorm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.2)
        cbar.set_label('Iteration')
        # plt.tight_layout()  # not compatible with tight_layout
        plt.savefig(self.save_folder + "mnist_activity_corr_dist_0-9.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Correlation cross correlation")

    def activity_measures(self):
        set_figure(4)
        test_point_indices = self.test_point_indices
        init, lr, bs, neurons = self.hyperparam
        ts = self.cth
        cfs = []
        conn_probs = []
        edges = []
        degrees = []
        forman_ricci_entropies = []
        t = ts[0]
        for s in test_point_indices:
            data = self.get_activity_measures(s=s)
            cfs.append(data["frobenius_norm"])
            conn_probs.append(data[f"connectivity_prob_cth{t}"])
            edges.append(data[f"number_of_edges_cth{t}"])
            degrees.append(data[f"degree_cth{t}"])
            forman_ricci_entropies.append(data[f"forman_ricci_entropy_cth{t}"])
        fig, ax = plt.subplots()
        cfs = np.array(cfs) / np.max(cfs)
        conn_probs = np.array(conn_probs) / np.max(conn_probs)
        ds = np.stack(degrees).mean(axis=1)
        ds = ds / np.max(ds)
        fre = np.array(forman_ricci_entropies) / np.max(forman_ricci_entropies)
        n = np.array(test_point_indices) / 100000
        ax.plot(n, cfs, lw=1, label=r"$||C||_F$", color=color_list[0])
        ax.plot(n, conn_probs, lw=1, label=r"$P(|\rho| > \rho_{th})$", color=color_list[1])
        ax.plot(n, ds, lw=1, label=r"$\langle k \rangle$", color=color_list[2])
        ax.plot(n, fre, lw=1, label=r"$H_{FR}$", color=color_list[3])
        ax.set_xlim([-0.1, 4])
        ax1 = fig.add_axes([0.6, 0.2, 0.5, 0.5])
        ax1.plot(test_point_indices, cfs, lw=1, label=r"$||C||_F$", color=color_list[0])
        ax1.plot(test_point_indices, conn_probs, lw=1, label=r"$P(|\rho| > \rho_{th})$", color=color_list[1])
        ax1.plot(test_point_indices, ds, lw=1, label=r"$\langle k \rangle$", color=color_list[2])
        ax1.plot(test_point_indices, fre, lw=1, label=r"$H_{FR}$", color=color_list[3])
        ax1.set_xscale('log')
        y0, y1 = ax1.get_ylim()
        ax1.vlines(x=np.argmax(cfs), ymin=y0 - 1, ymax=y1 + 1, linestyle='--', color='black')
        ax1.set_ylim([y0, y1])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax.set_xlabel(r"$t$")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(loc=(0.2, 0.8), frameon=False, ncol=2, fontsize=16)
        ax.set_xticks([0, 2, 4])
        ax.text(x=4.5, y=-0.1, s=r"$\times 10^5$", fontsize=16)
        plt.savefig(self.save_folder + "mnist_activity_measures.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("mnist_activity_measures")

    def NCG_property_model_size(self, data_file="./figures/phase_transition_model_size.npz"):
        set_figure(4)
        data = np.load(data_file)
        list_hyperparam = data['list_hyperparam']
        max_iter = self.max_iter
        ss = [i for i in range(0, 20)] + [i for i in range(20, 200, 10)] + [i for i in range(200, max_iter, 2000)] + [max_iter]

        def linear(x, a, b):
            return a * x + b

        fig, ax = plt.subplots()
        ms = []
        fns = []
        mcs = []
        ads = []
        for param in list_hyperparam:
            init, lr, bs, neurons = param
            init = int(init)
            lr = float(lr)
            bs = int(bs)
            neurons = int(neurons)
            if neurons > 2:
                frobenius_norms = data[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_frobenius_norm"]
                max_cluster_sizes = data[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_max_cluster_size"]
                average_degrees = data[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_average_degree"]
                average_finite_cluster_sizes = data[
                    f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_average_finite_cluster_sizes"]
                idx = ss.index(100)
                N = 2 * neurons + 10
                ms.append(N)
                fns.append(frobenius_norms[idx] / N ** 2)
                mcs.append(max_cluster_sizes[idx])
                ads.append(average_degrees[idx])
        ms = np.array(ms)
        sort_idx = np.argsort(ms)
        fns = np.array(fns)
        mcs = np.array(mcs)
        ads = np.array(ads)
        ms = ms[sort_idx]
        fns = fns[sort_idx]
        mcs = mcs[sort_idx]
        ads = ads[sort_idx]
        plt.plot(ms, fns, ls='none', lw=1, marker='.', ms=10, color=color_list[0], alpha=0.5, label=r"$||C||_F/M^2$")
        popt, pcov = curve_fit(linear, np.log(ms), np.log(fns))
        ax.plot(ms, np.exp(linear(np.log(ms), *popt)), lw=2, color=color_list[0])
        ax.text(x=200, y=0.1, s=f"{popt[0]:.2f}", color=color_list[0], fontsize=16)
        plt.plot(ms, mcs, ls='none', lw=1, marker='.', ms=10, color=color_list[1], alpha=0.5, label=r"$S_1$")
        popt, pcov = curve_fit(linear, np.log(ms), np.log(mcs))
        ax.text(x=200, y=40, s=f"{popt[0]:.2f}", color=color_list[1], fontsize=16)
        ax.plot(ms, np.exp(linear(np.log(ms), *popt)), lw=2, color=color_list[1])
        plt.plot(ms, ads, ls='none', lw=1, marker='.', ms=10, color=color_list[2], alpha=0.5, label=r"$\langle k \rangle$")
        popt, pcov = curve_fit(linear, np.log(ms), np.log(ads))
        ax.plot(ms, np.exp(linear(np.log(ms), *popt)), lw=2, color=color_list[2])
        ax.text(x=200, y=1.2, s=f"{popt[0]:.2f}", color=color_list[2], fontsize=16)
        ax.legend(loc='upper left', bbox_to_anchor=(0., 1.1), frameon=False, ncol=1, fontsize=14)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel(r"$M$")
        # ax.set_ylabel(r"$||C||_F/M$")
        ax.set_ylim([1e-2, 1e4])
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(f"./figures/connectivity_NCG_property_model_size.svg", dpi=600,
                    bbox_inches="tight", transparent=True)
        self.info(f"connectivity_NCG_property_model_size")

    def NCG_property_model_size_final_iter(self, data_file="./figures/NCG_property.npz"):
        set_figure(4)
        data = np.load(data_file)
        max_iter = 200000
        ss = [i for i in range(0, 20)] + [i for i in range(20, 200, 10)] + [i for i in range(200, max_iter, 2000)] + [max_iter]
        ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]

        def linear(x, a, b):
            return a * x + b

        fig, ax = plt.subplots()
        fns = []
        mcs = []
        ads = []
        for n in ns:
            init = 0
            lr = 0.1
            bs = 64
            N = 2 * n + 10
            frobenius_norms = data[f"init{init}_lr{lr}_bs{bs}_neurons{n}_frobenius_norm"]
            max_cluster_sizes = data[f"init{init}_lr{lr}_bs{bs}_neurons{n}_max_cluster_size"]
            average_degrees = data[f"init{init}_lr{lr}_bs{bs}_neurons{n}_average_degree"]
            fns.append(frobenius_norms[-1] / N ** 2)
            mcs.append(max_cluster_sizes[-1])
            ads.append(average_degrees[-1])

        fns = np.array(fns)
        mcs = np.array(mcs)
        ads = np.array(ads)
        plt.plot(ns, fns, ls='none', lw=1, marker='.', ms=10, color=color_list[0], alpha=0.5, label=r"$||C||_F/M^2$")
        popt, pcov = curve_fit(linear, np.log(ns), np.log(fns))
        ax.plot(ns, np.exp(linear(np.log(ns), *popt)), lw=2, color=color_list[0])
        ax.text(x=200, y=0.1, s=f"{popt[0]:.2f}", color=color_list[0], fontsize=16)
        plt.plot(ns, mcs, ls='none', lw=1, marker='.', ms=10, color=color_list[1], alpha=0.5, label=r"$S_1$")
        popt, pcov = curve_fit(linear, np.log(ns), np.log(mcs))
        ax.text(x=200, y=60, s=f"{popt[0]:.2f}", color=color_list[1], fontsize=16)
        ax.plot(ns, np.exp(linear(np.log(ns), *popt)), lw=2, color=color_list[1])
        plt.plot(ns, ads, ls='none', lw=1, marker='.', ms=10, color=color_list[2], alpha=0.5,
                 label=r"$\langle k \rangle$")
        popt, pcov = curve_fit(linear, np.log(ns), np.log(ads))
        ax.plot(ns, np.exp(linear(np.log(ns), *popt)), lw=2, color=color_list[2])
        ax.text(x=200, y=1.2, s=f"{popt[0]:.2f}", color=color_list[2], fontsize=16)
        ax.legend(loc='upper left', bbox_to_anchor=(0., 1.1), frameon=False, ncol=1, fontsize=14)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel(r"$M$")
        # ax.set_ylabel(r"$||C||_F/M$")
        ax.set_ylim([1e-2, 1e4])
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(f"./figures/connectivity_NCG_property_model_size_final_iter.svg", dpi=600,
                    bbox_inches="tight", transparent=True)
        self.info(f"connectivity_NCG_property_model_size_final_iter")

    def NCG_property_evolution_model_size(self, data_file="./figures/NCG_property.npz"):
        set_figure(3)
        data = np.load(data_file)
        max_iter = self.max_iter
        ss = [i for i in range(0, 20)] + [i for i in range(20, 200, 10)] + [i for i in range(200, max_iter, 2000)] + [max_iter]
        ns = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        keys = [
            'frobenius_norm',
            'max_cluster_size',
            'average_degree',
            'average_finite_cluster_sizes',
            'susceptibility',
        ]
        names = [
            r"$||C||_F/M^2$",
            r"$S_1/M$",
            r"$\langle k \rangle$",
            r"$\langle S_{>1} \rangle$",
            r"$\chi$",
        ]
        norms = [
            LogNorm(vmin=5, vmax=1000),
            LogNorm(vmin=30, vmax=1000),
            LogNorm(vmin=30, vmax=1000),
            LogNorm(vmin=2, vmax=1000),
            LogNorm(vmin=2, vmax=1000),
        ]
        nths = [2, 30, 10, 0, 0]
        cmap = plt.get_cmap("jet")
        for k, name, norm, nth in zip(keys, names, norms, nths):
            fig, ax = plt.subplots()
            for neurons in ns:
                if neurons > nth:
                    N = float(2 * neurons + 10)
                    init = 0
                    lr = 0.1
                    bs = 64
                    x = data[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_{k}"].astype(float)
                    if k == "frobenius_norm":
                        x /= N ** 2
                    if k == "max_cluster_size":
                        x /= N
                    color = cmap(norm(N))
                    ax.plot(ss, x, lw=1, color=color)
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # for older matplotlib versions
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
            cbar.set_label(r'$M$')
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(name)
            ax.spines[['right', 'top']].set_visible(False)
            plt.savefig(self.save_folder + f"{k}_process_model_size.svg", dpi=600,
                        bbox_inches="tight", transparent=True)
            plt.close()
            self.info(f"{k}_process_model_size")

    def NCG_property_evolution_batch_size(self, data_file="./figures/NCG_property.npz"):
        set_figure(3)
        data = np.load(data_file)
        max_iter = self.max_iter
        ss = [i for i in range(0, 20)] + [i for i in range(20, 200, 10)] + [i for i in range(200, max_iter, 2000)] + [max_iter]
        bs = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]

        keys = [
            'frobenius_norm',
            'max_cluster_size',
            'average_degree',
            'average_finite_cluster_sizes',
            'susceptibility',
        ]
        names = [
            r"$||C||_F/M^2$",
            r"$S_1/M$",
            r"$\langle k \rangle$",
            r"$\langle S_{>1} \rangle$",
            r"$\chi$",
        ]
        norms = [
            LogNorm(vmin=2, vmax=1024),
            LogNorm(vmin=2, vmax=1024),
            LogNorm(vmin=2, vmax=1024),
            LogNorm(vmin=2, vmax=1024),
            LogNorm(vmin=2, vmax=1024),
        ]
        nths = [8, 8, 8, 8, 8]
        cmap = plt.get_cmap("jet")
        for k, name, norm, nth in zip(keys, names, norms, nths):
            fig, ax = plt.subplots()
            for b in bs:
                if b > nth:
                    init = 0
                    lr = 0.1
                    neurons = 50
                    N = float(2 * neurons + 10)
                    x = data[f"init{init}_lr{lr}_bs{b}_neurons{neurons}_{k}"].astype(float)
                    if k == "frobenius_norm":
                        x /= N ** 2
                    if k == "max_cluster_size":
                        x /= N
                    color = cmap(norm(b))
                    ax.plot(ss, x, lw=1, color=color)

            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # for older matplotlib versions
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
            cbar.set_label(r'$B$')
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(name)
            ax.spines[['right', 'top']].set_visible(False)
            plt.savefig(self.save_folder + f"{k}_process_batch_size.svg", dpi=600,
                        bbox_inches="tight", transparent=True)
            plt.close()
            self.info(f"{k}_process_batch_size")

    def NCG_property_evolution_learning_rate(self, data_file="./figures/NCG_property.npz"):
        set_figure(3)
        data = np.load(data_file)
        max_iter = self.max_iter
        ss = [i for i in range(0, 20)] + [i for i in range(20, 200, 10)] + [i for i in range(200, max_iter, 2000)] + [max_iter]
        lrs = [3.20000000, 1.60000000, 1.06666667, 0.80000000, 0.53333333, 0.40000000, 0.26666667, 0.20000000,
               0.13333333, 0.10000000, 0.06666667, 0.05000000, 0.03333333, 0.02500000, 0.01666667, 0.01250000,
               0.00833333, 0.00625000, 0.00312500, 0.00156250, 0.00078125, 0.00039063, 0.00019531, 0.00009766]

        keys = [
            'frobenius_norm',
            'max_cluster_size',
            'average_degree',
            'average_finite_cluster_sizes',
            'susceptibility',
        ]
        names = [
            r"$||C||_F/M^2$",
            r"$S_1/M$",
            r"$\langle k \rangle$",
            r"$\langle S_{>1} \rangle$",
            r"$\chi$",
        ]
        norms = [
            LogNorm(vmin=0.005, vmax=2),
            LogNorm(vmin=0.005, vmax=2),
            LogNorm(vmin=0.005, vmax=2),
            LogNorm(vmin=0.005, vmax=2),
            LogNorm(vmin=0.005, vmax=2),
        ]
        nths = [2, 2, 2, 2, 2]
        cmap = plt.get_cmap("jet")
        for k, name, norm, nth in zip(keys, names, norms, nths):
            fig, ax = plt.subplots()
            for lr in lrs[::-1]:
                if lr < nth:
                    init = 0
                    bs = 64
                    neurons = 50
                    N = float(2 * neurons + 10)
                    x = data[f"init{init}_lr{lr}_bs{bs}_neurons{neurons}_{k}"].astype(float)
                    if k == "frobenius_norm":
                        x /= N ** 2
                    if k == "max_cluster_size":
                        x /= N
                    color = cmap(norm(lr))
                    ax.plot(ss, x, lw=1, color=color)

            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # for older matplotlib versions
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
            cbar.set_label(r'$\upsilon$')
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(name)
            ax.spines[['right', 'top']].set_visible(False)
            plt.savefig(self.save_folder + f"{k}_process_learning_rate.svg", dpi=600,
                        bbox_inches="tight", transparent=True)
            plt.close()
            self.info(f"{k}_process_learning_rate")

    def NCG_property_evolution_mobility_factor(self, data_file="./figures/NCG_property.npz"):
        set_figure(3)
        data = np.load(data_file)
        max_iter = self.max_iter
        ss = [i for i in range(0, 20)] + [i for i in range(20, 200, 10)] + [i for i in range(200, max_iter, 2000)] + [max_iter]
        bs = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
        lrs = [3.20000000, 1.60000000, 1.06666667, 0.80000000, 0.53333333, 0.40000000, 0.26666667, 0.20000000,
               0.13333333, 0.10000000, 0.06666667, 0.05000000, 0.03333333, 0.02500000, 0.01666667, 0.01250000,
               0.00833333, 0.00625000, 0.00312500, 0.00156250, 0.00078125, 0.00039063, 0.00019531, 0.00009766]
        mfs = []
        for b in bs:
            for lr in lrs:
                mfs.append((b, lr, np.log(lr / b)))
        mfs = sorted(mfs, key=lambda x: x[2])

        keys = [
            'frobenius_norm',
            'max_cluster_size',
            'average_degree',
            'average_finite_cluster_sizes',
            'susceptibility',
        ]
        names = [
            r"$||C||_F/M^2$",
            r"$S_1/M$",
            r"$\langle k \rangle$",
            r"$\langle S_{>1} \rangle$",
            r"$\chi$",
        ]
        norms = [
            Normalize(vmin=-9, vmax=-5),
            Normalize(vmin=-9, vmax=-5),
            Normalize(vmin=-9, vmax=-5),
            Normalize(vmin=-9, vmax=-5),
            Normalize(vmin=-9, vmax=-5),
        ]
        cmap = plt.get_cmap("jet")
        for k, name, norm in zip(keys, names, norms):
            fig, ax = plt.subplots()
            for mf in mfs:
                if -5.1 > mf[2] > -9.9:
                    init = 0
                    b = mf[0]
                    lr = mf[1]
                    neurons = 50
                    N = float(2 * neurons + 10)
                    try:
                        x = data[f"init{init}_lr{lr}_bs{b}_neurons{neurons}_{k}"].astype(float)
                    except KeyError:
                        continue
                    if k == "frobenius_norm":
                        x /= N ** 2
                    if k == "max_cluster_size":
                        x /= N
                    color = cmap(norm(mf[2]))
                    ax.plot(ss, x, lw=1, color=color)

            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # for older matplotlib versions
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
            cbar.set_label(r'$\ln \gamma$')
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(name)
            ax.spines[['right', 'top']].set_visible(False)
            plt.savefig(self.save_folder + f"{k}_process_mobility_factor.svg", dpi=600,
                        bbox_inches="tight", transparent=True)
            plt.close()
            self.info(f"{k}_process_mobility_factor")

    def NCG_degree_component_clique_dist(self, data_file="./figures/NCG_property.npz", aim="degree"):
        set_figure(3)
        data = np.load(data_file)
        max_iter = self.max_iter
        ss = [i for i in range(0, 20)] + [i for i in range(20, 200, 10)] + [i for i in range(200, max_iter, 2000)] + [max_iter]
        ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        cmap = self.iter_cmap
        lognorm = self.lognorm
        name = {
            "degree@x": r"$k$",
            "component@x": r"$S$",
            "clique@x": r"$S_C$",
            "clique_value@x": r"$a_C$",
            "degree@y": r"$p(k)$",
            "component@y": r"$p(S)$",
            "clique@y": r"$p(S_C)$",
            "clique_value@y": r"$p(a_C)$",
        }

        for n in ns:
            fig, ax = plt.subplots()
            init = 0
            lr = 0.1
            bs = 64
            for s in ss:
                e = data[f"init{init}_lr{lr}_bs{bs}_neurons{n}_{aim}_hist_e_iter{s}"]
                p = data[f"init{init}_lr{lr}_bs{bs}_neurons{n}_{aim}_hist_p_iter{s}"]
                color = cmap(lognorm(s))
                ax.plot(e, p, lw=1, color=color)
            sm = ScalarMappable(cmap=cmap, norm=lognorm)
            sm.set_array([])  # for older matplotlib versions
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
            cbar.set_label(r'$\ln t$')
            cbar.set_ticks([])
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel(name[f"{aim}@x"])
            ax.set_ylabel(name[f"{aim}@y"])
            ax.spines[['right', 'top']].set_visible(False)
            plt.savefig(self.save_folder + f"NCG_{aim}_dist_neurons{n}.svg", dpi=600,
                        bbox_inches="tight", transparent=True)
            plt.close()
            self.info(f"NCG_{aim}_dist_neurons{n}")

    def NCG_degree_dist_compare(self, data_file="./figures/NCG_property.npz", iter=-1):
        set_figure(4)
        data = np.load(data_file)
        max_iter = 200000
        ss = [i for i in range(0, 20)] + [i for i in range(20, 200, 10)] + [i for i in range(200, max_iter, 2000)] + [max_iter]
        ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        cmap = plt.get_cmap("jet")
        norm = Normalize(vmin=2 * ns[0] + 10, vmax=2 * ns[-1] + 10)

        def linear(x, a, b):
            return a * x + b

        es = []
        ps = []
        fig, ax = plt.subplots()
        for n in ns:
            init = 0
            lr = 0.1
            bs = 64
            N = 2 * n + 10
            color = cmap(norm(N))
            if iter == -1:
                s = ss[-1]
            else:
                s = iter
            e = data[f"init{init}_lr{lr}_bs{bs}_neurons{n}_degree_hist_e_iter{s}"]
            p = data[f"init{init}_lr{lr}_bs{bs}_neurons{n}_degree_hist_p_iter{s}"]
            ax.plot(e, p, ls='none', marker='.', ms=10, color=color, alpha=0.5)
            es.append(e)
            ps.append(p)
            # ax.plot(x, np.exp(linear(np.log(x), *popt)), lw=1, color='black')
        es = np.concatenate(es, axis=0)
        ps = np.concatenate(ps, axis=0)
        x = []
        y = []
        for e in np.unique(es):
            x.append(e)
            y.append(np.mean(ps[es == e]))
        # ax.plot(x, y, lw=3, color='black')
        x = np.array(x)
        y = np.array(y)
        idx = x > 1
        x = x[idx]
        y = y[idx]
        popt, pcov = curve_fit(linear, np.log(x), np.log(y))
        ax.plot(x, np.exp(linear(np.log(x), *popt)), lw=3, color='black')
        if iter == -1:
            xc = 0.05
        elif iter == 100:
            xc = 0.1
        else:
            xc = 0.75
        ax.text(x=xc, y=1e-3, s=rf"$\alpha = {-popt[0]:.2f}$", color='black', fontsize=16)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$M$')
        cbar.set_ticks([])
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$p(k)$")
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + f"NCG_degree_dist_{iter}.svg", dpi=600,
                    bbox_inches="tight", transparent=True)
        plt.close()
        self.info(f"NCG_degree_dist_{iter}")

    def NCG_clique_dist_compare(self, data_file="./figures/NCG_property.npz", iter=-1):
        set_figure(4)
        data = np.load(data_file)
        max_iter = 200000
        ss = [i for i in range(0, 20)] + [i for i in range(20, 200, 10)] + [i for i in range(200, max_iter, 2000)] + [max_iter]
        ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        cmap = plt.get_cmap("jet")
        norm = Normalize(vmin=2 * ns[0] + 10, vmax=2 * ns[-1] + 10)

        def func(s, a, b, c, d):
            # return np.log(a * np.exp(-x**2/b) + x**(-c))
            eps = 1e-12  # tune as needed
            x = a * (s + eps) ** (-b) * np.exp(-((s / c) ** d) + 1e-10)
            out = np.full_like(x, np.nan, dtype=float)
            mask = np.isfinite(x) & (x > 0)
            out[mask] = np.log(x[mask] + eps)
            return out

        def linear(x, a, b):
            return a * x + b

        fig, ax = plt.subplots()
        es = []
        ps = []
        for n in ns:
            init = 0
            lr = 0.1
            bs = 64
            N = 2 * n + 10
            color = cmap(norm(N))
            if iter == -1:
                s = ss[-1]
            else:
                s = iter
            e = data[f"init{init}_lr{lr}_bs{bs}_neurons{n}_clique_hist_e_iter{s}"]
            p = data[f"init{init}_lr{lr}_bs{bs}_neurons{n}_clique_hist_p_iter{s}"]
            ax.plot(e, p, ls='none', marker='.', ms=10, color=color)
            es.append(e)
            ps.append(p)
        es = np.concatenate(es, axis=0).astype(int)
        ps = np.concatenate(ps, axis=0)
        x = []
        y = []
        for e in np.unique(es):
            x.append(e)
            y.append(np.mean(ps[es == e]))
        # ax.plot(x, y, lw=3, color='black')
        x = np.array(x)
        y = np.array(y)
        # idx = x < 16
        # x = x[idx]
        # y = y[idx]
        # popt, pcov = curve_fit(func, x, np.log(y))
        # ax.plot(x, np.exp(func(x, *popt)), lw=3, color='black')
        idx = x > 4
        x = x[idx]
        y = y[idx]
        popt, pcov = curve_fit(linear, np.log(x), np.log(y))
        ax.plot(x, np.exp(linear(np.log(x), *popt)), lw=3, color='black')
        if iter == -1:
            yc = 5e-4
        elif iter == 100:
            yc = 2e-3
        else:
            yc = 1e-3
        ax.text(x=1.1, y=yc, s=rf"$\alpha = {popt[1]:.2f}$", color='black', fontsize=16)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$M$')
        cbar.set_ticks([])
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel(r"$S_C$")
        ax.set_ylabel(r"$p(S_C)$")
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + f"NCG_clique_dist_iter{iter}.svg", dpi=600,
                    bbox_inches="tight", transparent=True)
        plt.close()
        self.info(f"NCG_clique_dist_iter{iter}")

    def NCG_clique_value_compare(self, data_file="./figures/NCG_property.npz"):
        set_figure(4)
        data = np.load(data_file)
        max_iter = 200000
        ss = [i for i in range(0, 20)] + [i for i in range(20, 200, 10)] + [i for i in range(200, max_iter, 2000)] + [
            max_iter]
        ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        cmap = plt.get_cmap("jet")
        norm = Normalize(vmin=2 * ns[0] + 10, vmax=2 * ns[-1] + 10)

        def linear(x, a, b):
            return a * x + b

        fig, ax = plt.subplots()
        es = []
        ps = []
        for n in ns:
            init = 0
            lr = 0.1
            bs = 64
            N = 2 * n + 10
            color = cmap(norm(N))
            s = ss[-1]
            e = data[f"init{init}_lr{lr}_bs{bs}_neurons{n}_clique_value_hist_e_iter{s}"]
            p = data[f"init{init}_lr{lr}_bs{bs}_neurons{n}_clique_value_hist_p_iter{s}"]
            ax.plot(e, p, ls='none', marker='.', ms=10, color=color, alpha=0.5)
            if n > 300:
                es.append(e)
                ps.append(p)
        es = np.concatenate(es, axis=0).astype(int)
        ps = np.concatenate(ps, axis=0)
        x = []
        y = []
        for e in np.unique(es):
            x.append(e)
            y.append(np.mean(ps[es == e]))
        x = np.array(x)
        y = np.array(y)
        idx = x > 6
        popt, pcov = curve_fit(linear, np.log(x[idx]), np.log(y[idx]))
        x_new = np.linspace(5, 20, 100)
        ax.plot(x_new, np.exp(linear(np.log(x_new), *popt)), lw=3, color='black')
        ax.text(x=0.2, y=3e-4, s=rf"$\alpha = {-popt[0]:.2f}$", color='black', fontsize=16)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$M$')
        cbar.set_ticks([])
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel(r"$a_C$")
        ax.set_ylabel(r"$p(a_C)$")
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + f"NCG_clique_value_dist.svg", dpi=600,
                    bbox_inches="tight", transparent=True)
        plt.close()
        self.info(f"NCG_clique_value_dist")

    def loss_acc(self):
        set_figure(4)
        data = np.load(self.path_file)
        loss = data['losses']
        acc = data['accuracy']
        fig, ax = plt.subplots()
        ax.plot(loss, color=color_list[0], lw=1, alpha=1)
        ax1 = plt.twinx(ax)
        test_point_indices = self.test_point_indices
        ax1.plot(test_point_indices[1:], (acc + 1e-8) / 100, lw=1, alpha=1, color=color_list[1])
        cfs = []
        for s in test_point_indices:
            data = self.get_activity_measures(s=s)
            cfs.append(data["frobenius_norm"])
        cfs = np.array(cfs)
        ax1.plot(test_point_indices, cfs / np.max(cfs), lw=1, alpha=1, color=color_list[2], label=r"$||C||_F$")
        plt.legend(loc='lower right', frameon=False, fontsize=16)
        ax.set_yscale('log')
        # ax1.set_yscale('log')
        plt.xscale('log')
        ax.vlines(x=np.argmax(cfs), ymin=-0.5, ymax=10, color='black', linestyle="--", lw=2)
        ax.set_ylim([1e-10, 10])
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$\langle L \rangle$', color=color_list[0])
        ax1.set_ylabel('Accuracy', color=color_list[1])
        plt.savefig(self.save_folder + "mnist_loss_acc.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Loss accuracy")

    def degree_dist(self):
        fig, ax = plt.subplots()
        cmap = self.iter_cmap
        lognorm = self.lognorm
        for s in self.test_point_indices:
            data = self.get_activity_key_measures(s=s, treat_negatives='abs', use_fisher_z=False, keep_activity_files=True)
            cth = data['correlation_threshold']
            D = data[f"degree_cth{cth}"]
            p, e = hist(D, bins=50)
            color = cmap(lognorm(s + 1))
            ax.plot(e, p, color=color, ls='-', marker='.', ms=5, lw=1, alpha=0.3)
        sm = ScalarMappable(cmap=cmap, norm=lognorm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$\ln t$')
        cbar.set_ticks([])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + "degree_dist.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Degree distribution")

    def NCG_network_properties(self):
        cth = 0.5
        keys = [
            'largest_component_fraction',
            'susceptibility',
            'global_efficiency',
            'small_worldness_sigma',
            'small_worldness_omega',
            'clustering_coefficient',
            'heterogeneity',
            'spectral_radius',
            'algebraic_connectivity',
            'kcore_max_k',
            'SIS_indicator',
            'kuramoto_Kc',
            'diffusion_mixing_time',
            'branching_ratio_proxy',
            f'connectivity_prob_cth{cth}',
            f"number_of_edges_cth{cth}",
            f"degree_cth{cth}",
            f"forman_ricci_entropy_cth{cth}",
            # f"forman_ricci_curvatures_cth{cth}",
            f"cluster_sizes_cth{cth}",
            f"cluster_values_cth{cth}",
            f"clique_sizes_cth{cth}",
            f"clique_values_cth{cth}",
        ]
        names = [
            r"$S_1/M$",
            r"$\chi$",
            r"$E_{glo}$",
            r"$\sigma_{sw}$",
            r"$\omega_{sw}$",
            r"$\sigma_{cc}$",
            r"$\sigma_{heter}$",
            r"$\sigma_{rad}$",
            r"$\sigma_{alg}$",
            r"$k_{core}$",
            r"$\sigma_{SIS}$",
            r"$k_c$",
            r"$\tau_{diff}$",
            r"$\sigma_{branch}$",
            r"$P(|\rho| > \rho_{th})$",
            r"$\langle N_E \rangle$",
            r"$\langle k \rangle$",
            r"$H_{FR}$",
            r"$\langle S_1 \rangle$",
            r"$\langle V_1 \rangle$",
            r"$\langle S_C \rangle$",
            r"$\langle V_C \rangle$",
        ]
        key_list = [
            f"number_of_edges_cth{cth}",
            f"degree_cth{cth}",
            f"cluster_sizes_cth{cth}",
            f"cluster_values_cth{cth}",
            f"clique_sizes_cth{cth}",
            f"clique_values_cth{cth}",
        ]
        set_figure(4)
        for key, name in zip(keys, names):
            fig, ax = plt.subplots()
            obs = []
            for s in self.test_point_indices:
                data = self.get_activity_key_measures(s=s, treat_negatives='abs', use_fisher_z=False, keep_activity_files=True)
                val = data[key]
                if any(key.startswith(k) for k in key_list):
                    obs.append(np.mean(val))
                else:
                    obs.append(val)
            ax.plot(self.test_point_indices, obs, lw=1, label=key, color='black')
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(name)
            ax.spines[['right', 'top']].set_visible(False)
            plt.savefig(self.save_folder + f"{key}.svg",
                        dpi=600, bbox_inches="tight", transparent=True)
            plt.close()
            self.info(f"{key}")

    def clique_dist(self):
        set_figure(3)
        cmap = self.iter_cmap
        fig, axes = plt.subplots(5, 3, figsize=(3.5 * 3, 3.5 * 3))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        ts = np.linspace(0.5, 1, 10)
        init, lr, bs, neurons = self.hyperparam
        norm_cth = Normalize(vmin=ts.min(), vmax=ts.max())
        for i, s in enumerate([0, 100, 200000]):
            data1 = self.get_activity_measures(s=s)
            data = self.get_activity_measures_minibatch(s=s)
            for t in ts:
                ds = data1[f"degree_cth{t}"]
                cluster_values = data[f"cluster_values_cth{t}"]
                clique_values = data[f"clique_values_cth{t}"]
                cluster_sizes = data[f"cluster_sizes_cth{t}"]
                clique_sizes = data[f"clique_sizes_cth{t}"]
                color = cmap(norm_cth(t))
                bins = int(np.max(ds)) if np.max(ds) > 0 else 10
                p, e = hist(ds.reshape(-1), bins=bins)
                axes[0, i].plot(e, p, lw=1, alpha=1, color=color)
                axes[0, i].set_xlabel("Connection degree")
                axes[0, i].set_xscale('log')
                axes[0, i].set_yscale('log')
                axes[0, i].set_xlim([0.9, 13])
                p, e = hist(cluster_sizes, bins=30)
                axes[1, i].plot(e, p, lw=1, alpha=1, color=color)
                axes[1, i].set_xlabel("cluster size")
                axes[1, i].set_xscale('log')
                axes[1, i].set_yscale('log')
                axes[1, i].set_xlim([0.9, 100])
                p, e = hist(clique_sizes, bins=30)
                axes[2, i].plot(e, p, lw=1, alpha=1, color=color)
                axes[2, i].set_xlabel("Clique size")
                axes[2, i].set_xscale('log')
                axes[2, i].set_yscale('log')
                axes[2, i].set_xlim([0.9, 13])
                p, e = hist(cluster_values, bins=30)
                axes[3, i].plot(e, p, lw=1, alpha=1, color=color)
                axes[3, i].set_xlabel("Cluster value")
                axes[3, i].set_xscale('log')
                axes[3, i].set_yscale('log')
                axes[3, i].set_xlim([0.9, 2e4])
                p, e = hist(clique_values, bins=30)
                axes[4, i].plot(e, p, lw=1, alpha=1, color=color)
                axes[4, i].set_xlabel("Clique values")
                axes[4, i].set_xscale('log')
                axes[4, i].set_yscale('log')
                axes[4, i].set_xlim([0.9, 2e3])

        sm = ScalarMappable(cmap=cmap, norm=norm_cth)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=axes, shrink=0.5)
        cbar.set_label(r'$C_{th}$')
        fig.supylabel("Probability density")
        plt.savefig(self.save_folder + "mnist_activity_corr_dist_minibatch_cth.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Clique size distribution")

    def pred_dist(self):
        set_figure(4)
        cmap = self.iter_cmap
        lognorm = self.lognorm
        init, lr, bs, neurons = self.hyperparam
        fig, ax = plt.subplots()
        iters = self.test_point_indices
        for s in iters[:100] + iters[-5:]:
            data = self.get_activity_measures(s=s)
            p = data["pred_hist_prob"]
            e = data["pred_hist_value"]
            color = cmap(lognorm(s + 1))
            plt.plot(e, p, lw=1, label=f"iter {s}", color=color, alpha=0.8)

        sm = ScalarMappable(cmap=cmap, norm=lognorm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$\ln t$')
        cbar.set_ticks([])
        plt.yscale('log')
        # plt.xscale('log')
        plt.xlabel(r"$a$")
        plt.ylabel(r"p(a)")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(self.save_folder + "mnist_pred_dist.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("mnist_pred_dist")

    def pred_entropy(self):
        set_figure(4)
        fig, ax = plt.subplots()
        entropies = []
        init, lr, bs, neurons = self.hyperparam
        iters = self.test_point_indices
        for s in iters:
            data = self.get_activity_measures(s=s)
            entropies.append(data["pred_entropy"])
        iters = np.array(iters)
        ax.plot(iters / 100000, entropies, lw=1, color='black')
        ax1 = fig.add_axes([0.35, 0.35, 0.5, 0.5])
        ax1.plot(iters, entropies, lw=1, color='black')
        ax1.set_xscale('log')
        ax1.set_yticks([])
        y0, y1 = ax1.get_ylim()
        ax1.vlines(x=60, ymin=y0, ymax=y1, ls='--', color='red')
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$S_{pred}$")  # Prediction entropy
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.text(2.2, 0.3, r"$\times 10^5$", fontsize=16)
        plt.savefig(self.save_folder + "mnist_pred_entropy.svg",
                    dpi=300, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("mnist_pred_entropy")

    def complexity_pred_entropy(self):
        set_figure(4)
        fig, ax = plt.subplots()
        entropies = []
        cfs = []
        iters = self.test_point_indices
        init_type, lr, bs, neurons = self.hyperparam
        cmap = self.iter_cmap
        lognorm = self.lognorm
        for s in iters:
            file = self.save_folder + f"activity_measure_iter{s}.npz"
            data = self.get_activity_measures(s=s)
            entropies.append(data["pred_entropy"])
            cfs.append(data["frobenius_norm"])
        cfs = np.array(cfs)
        s0 = entropies[1]
        for i in range(1, len(cfs)):
            color = cmap(lognorm(iters[i] + 1))
            plt.plot(s0 - entropies[i], cfs[i] / 1000, color=color, ls='none', marker='o', markersize=10)
        plt.plot(s0 - entropies[1:], cfs[1:] / 1000, lw=1, color='black')  # initial state is excluded
        sm = ScalarMappable(cmap=cmap, norm=lognorm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$\ln t$')
        cbar.set_ticks([])
        plt.ylabel(r"$||C||_F$")
        plt.xlabel(r"$\Delta I$")  # Information gain
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.text(0, 1, r"$\times 10^3$", fontsize=16)
        plt.savefig(self.save_folder + "mnist_pred_entropy_frobenius_norm.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("mnist_pred_entropy_frobenius_norm")

    def sparsity_pred_entropy(self):
        set_figure(4)
        fig, ax = plt.subplots()
        iters = self.test_point_indices
        init, lr, bs, neurons = self.hyperparam
        cmap = self.iter_cmap
        lognorm = self.lognorm
        if not self.alphas:
            self.activity_power_law_zoom()
        entropies = []
        for s in iters:
            data = self.get_activity_measures(s=s)
            entropies.append(data["pred_entropy"])
        for i in range(1, len(iters)):
            color = cmap(lognorm(iters[i] + 1))
            ax.plot(self.alphas[i], entropies[i], color=color, ls='none', marker='o', markersize=3)
        ax1 = fig.add_axes([0.4, 0.5, 0.33, 0.33])
        for i in range(50, len(iters)):
            color = cmap(lognorm(iters[i] + 1))
            ax1.plot(self.alphas[i], entropies[i], color=color, ls='none', marker='o', markersize=2)
        ax1.set_yticks([])
        ax1.set_xticks([])
        # sns.regplot(x=alphas[1:100], y=entropies[1:100], scatter=False, line_kws=dict(color="r"))  # exclude the first point, which is not being tunned by backprop
        sm = ScalarMappable(cmap=cmap, norm=lognorm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$\ln t$')
        cbar.set_ticks([])
        ax.set_xlabel("Sparsity")
        ax.set_ylabel(r"$S_{pred}$")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(self.save_folder + "mnist_pred_entropy_sparsity.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("mnist_pred_entropy_sparsity")

    def raw_activities(self):
        set_figure(4)
        fig, ax = plt.subplots()
        iters = self.test_point_indices
        init, lr, bs, neurons = self.hyperparam
        cmap = self.iter_cmap
        lognorm = self.lognorm
        for s in iters:
            data = self.get_activity_measures(s=s)
            p = data["hist_prob"]
            e = data["hist_value"]
            color = cmap(lognorm(s + 1))
            plt.plot(e ** 2 / 1e7, p, lw=1, label=f"iter {s}", color=color, alpha=0.8)

        sm = ScalarMappable(cmap=cmap, norm=lognorm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$\ln t$')
        cbar.set_ticks([])
        plt.yscale('log')
        # plt.xscale('log')
        plt.xlabel(r"$a^2$")
        plt.ylabel(r"$p(a^2)$")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.text(1.4, 3e-8, r"$\times 10^7$", fontsize=16)
        plt.savefig(self.save_folder + "mnist_raw_activity_dist.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("mnist_raw_activity_dist")

    def num_NCG_edges_minibatch(self):
        set_figure(4)
        fig, ax = plt.subplots()
        cmap = self.iter_cmap
        lognorm = self.lognorm
        for si, s in enumerate([0, 100, 200000]):
            spks = self.get_spks(s=s)['spks']
            sizes = []
            for _ in range(1000):
                indices = np.random.randint(0, spks.shape[1], size=(64,))
                minibatch = spks[:, indices]
                corr = np_pearson_corr(minibatch, minibatch)
                conn_mat = np.abs(corr) >= 0.5
                sizes.append(np.sum(conn_mat))
            color = cmap(lognorm(s + 1))
            sizes = np.array(sizes)
            h = plt.hist(sizes / 1000, bins=30, density=True, alpha=0.8, color=color)
        sm = ScalarMappable(cmap=cmap, norm=lognorm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$\ln t$')
        cbar.set_ticks([])
        plt.xlabel(r"$N_E$")  # Number of edges
        plt.ylabel(r"$p(N_E)$")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.text(1.8, -1, r"$\times 10^3$", fontsize=16)
        plt.savefig(self.save_folder + "mnist_activity_corr_num_edges_dist_minibatch.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("mnist_activity_corr_num_edges_dist_minibatch")

    def clique_dist_fit(self):
        set_figure(4)
        s = self.max_iter
        cmap = self.iter_cmap
        init, lr, bs, neurons = self.hyperparam
        data = self.get_activity_measures_minibatch(s=s)
        ts = np.linspace(0.5, 1, 10)
        norm01 = Normalize(vmin=ts.min(), vmax=ts.max())
        fig, ax = plt.subplots(figsize=(4, 3.5))

        def func(s, a, b, c, d):
            # return np.log(a * np.exp(-x**2/b) + x**(-c))
            x = a * s ** (-b) * np.exp(-(s / c) ** d)
            return np.log(x + 1e-10)

        for icount, t in enumerate(ts):
            clique_sizes = data[f"clique_sizes_cth{t}"]
            p, e = hist(clique_sizes, bins=30)
            p0 = [1, 2, 3, 2]
            popt, pcov = curve_fit(func, e, np.log(p), p0=p0, bounds=((0, 0, 0, 0), (10, 20, 30, 20)))
            # print(*popt)
            color = cmap(norm01(t))
            plt.plot(e, p, lw=1, color=color, alpha=1, linestyle='-')
            xx = np.linspace(1, np.max(e), 100)
            plt.plot(xx, np.exp(func(xx, *popt)), lw=1, color='black', linestyle='--')
        sm = ScalarMappable(cmap=cmap, norm=norm01)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$C_{th}$')
        cbar.set_ticks([])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r"$S_C$")  # Clique size
        plt.ylabel(r"$p(S_C)$")
        plt.xticks([1, 10])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(self.save_folder + "mnist_activity_corr_component_size_dist_fit.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Clique size distribution fit to finite-size invariant distribution")

    def component_size_cth(self):
        set_figure(4)
        fig, ax = plt.subplots()
        ts = self.cth
        iters = self.test_point_indices
        init, lr, bs, neurons = self.hyperparam
        cmap = self.iter_cmap
        lognorm = self.lognorm
        for si, s in enumerate(iters):
            data = self.get_activity_measures(s=s)
            largest_sizes = []
            for t in ts:
                largest_sizes.append(np.max(data[f"cluster_sizes_cth{t}"]))
            color = cmap(lognorm(s + 1))
            plt.plot(ts, largest_sizes, lw=1, color=color)

        sm = ScalarMappable(cmap=cmap, norm=lognorm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$\ln t$')
        cbar.set_ticks([])
        plt.xlabel(r"$C_{th}$")
        plt.ylabel(r"$S_1$")  # Largest cluster size
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(self.save_folder + "mnist_activity_corr_th_clique_size.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("mnist_activity_corr_th_clique_size")

    def component_size_iter(self):
        set_figure(4)
        fig, ax = plt.subplots()
        ts = self.cth
        iters = self.test_point_indices
        init, lr, bs, neurons = self.hyperparam
        cmap = self.iter_cmap
        norm_ts = Normalize(vmin=ts.min(), vmax=ts.max())
        for t in ts:
            largest_sizes = []
            for si, s in enumerate(iters):
                data = self.get_activity_measures(s=s)
                largest_sizes.append(np.max(data[f"cluster_sizes_cth{t}"]))
            color = cmap(norm_ts(t))
            plt.plot(iters, largest_sizes, lw=1, color=color)

        sm = ScalarMappable(cmap=cmap, norm=norm_ts)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$C_{th}$')
        cbar.set_ticks([])
        plt.xlabel(r"$t$")
        plt.ylabel(r"$S_1$")  # Largest cluster size
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.savefig(self.save_folder + "mnist_activity_corr_th_clique_size_iter.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("mnist_activity_corr_th_clique_size_iter")

    def num_NCG_edges(self):
        set_figure(4)
        fig, ax = plt.subplots()
        ts = self.cth
        iters = self.test_point_indices
        init, lr, bs, neurons = self.hyperparam
        cmap = self.iter_cmap
        lognorm = self.lognorm
        for si, s in enumerate(iters):
            data = self.get_activity_measures(s=s)
            num_edges = []
            for t in ts:
                num_edges.append(data[f"number_of_edges_cth{t}"])
            color = cmap(lognorm(s + 1))
            plt.plot(ts, num_edges, lw=1, color=color)

        sm = ScalarMappable(cmap=cmap, norm=lognorm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$\ln t$')
        cbar.set_ticks([])
        plt.xlabel(r"$C_{th}$")
        plt.ylabel(r"$N_E$")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(self.save_folder + "mnist_activity_corr_th_edges.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("mnist_activity_corr_th_edges")


    def variance_corr_hebbin(self, cth=0.5):
        set_figure(4)
        path = self.path_data
        pcs = self.get_critical_connections(cth=cth, s=self.max_iter)['pcs']
        fig = plt.figure()
        ax1 = fig.add_axes((0.1, 0.82, 1, 0.18), xticklabels=[])
        ax2 = fig.add_axes((0.1, 0.1, 1, 0.7))
        x1 = np.var(path, axis=0)
        x2 = path[-1]
        x3 = path[-1] - path[0]
        x1 = np.abs(x1)
        x2 = np.abs(x2)
        x3 = np.abs(x3)
        coords = []
        for pc in pcs:
            coord = np.var(path[:, pc], axis=0) / x1.max()
            ax2.vlines(x=coord, ymin=1e-5, ymax=1, ls='-', lw=0.3, color='black', alpha=1)
            coords.append(coord)

        for pc in pcs:
            coord = path[-1, pc] / x2.max()
            coord = np.abs(coord)
            ax2.vlines(x=coord, ymin=1e-5, ymax=1, ls='-', lw=0.3, color='black', alpha=1)
            coords.append(coord)

        for pc in pcs:
            coord = (path[-1, pc] - path[0, pc]) / x3.max()
            coord = np.abs(coord)
            ax2.vlines(x=coord, ymin=1e-5, ymax=1, ls='-', lw=0.3, color='black', alpha=1)
            coords.append(coord)

        p, e = hist(x1, bins=1000)
        e = e / x1.max()
        ax2.plot(e, p, color=color_list[0], lw=2, alpha=0.9, label=r"$Var(w_t)$")
        p, e = hist(x2, bins=1000)
        e = e / x2.max()
        ax2.plot(e, p, color=color_list[1], lw=2, alpha=0.9, label=r"$w_T$")
        p, e = hist(x3, bins=1000)
        e = e / x3.max()
        ax2.plot(e, p, color=color_list[2], lw=2, alpha=0.9, label=r"$w_t - w_0$")

        # p, e = hist(coords, bins=20)
        # ax1 = ax.twinx()
        # ax1.plot(e, p, color='red', lw=2)
        # p, e = hist(np.log(coords), bins=20)
        # ax1.plot(e, p, lw=2, color='black')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        xlims = ax2.get_xlim()
        h = ax1.hist(np.log(coords), bins=20, histtype='stepfilled', color='gray', density=True)
        ax1.set_xlim([np.log(xlims[0]), np.log(xlims[1])])
        ax1.set_xticks([])
        ax1.set_yticks([0.5])

        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=3, fontsize=16)
        ax2.set_ylabel(r"$p$")
        plt.savefig(self.save_folder + "variance_corr.svg", dpi=600,
                    bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Variance, absolute value, shift of parameter with correlation (Hebbian plasticity)")

    def trajectory_autocorr(self, cth=0.5):
        set_figure(4)
        fig = plt.figure()
        points = []
        ax1 = fig.add_axes((0.1, 0.1, 0.7, 0.8))
        ax2 = fig.add_axes((0.82, 0.1, 0.18, 0.8))
        pcs = self.get_critical_connections(cth=cth, s=self.max_iter)['pcs']
        path = self.path_data
        test_point_indices = self.test_point_indices
        for pc in pcs:
            ax1.plot(test_point_indices, path[:, pc], lw=0.5, color=color_list[1], alpha=0.5)
            points.append(path[-1, pc])
        ax2.hist(points, bins=20, orientation='horizontal', color=color_list[1], alpha=0.5, density=True)
        indices = np.random.randint(0, 784 * 50 + 50 + 50 * 50 + 50 + 50 * 10 + 10, size=(1000,))
        points = []
        for i in indices:
            if i not in pcs:
                ax1.plot(test_point_indices, path[:, i], lw=0.5, color='black', alpha=0.5)
                points.append(path[-1, i])
        ax2.hist(points, bins=20, orientation='horizontal', color='black', alpha=0.5, density=True)
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_yticks([])
        ax2.set_xticks([5])
        ax1.set_xscale('log')
        ax1.set_xlabel(r"$t$")
        ax1.set_ylabel(r"$w$")
        ax1.set_xlim([1, 200000])
        ax1.set_xticks([1, 1000, 100000])
        # plt.tight_layout()
        plt.savefig(self.save_folder + "trajectories_corr.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Trajectory autocorrelation")

    def trajectory_irreversibility(self, cth=0.5):
        set_figure(4)
        fig = plt.figure()
        ax1 = fig.add_axes((0.1, 0.1, 0.9, 0.9))
        ax2 = fig.add_axes((0.8, 0.5, 0.47, 0.47))
        irs = []
        t = np.array(self.test_point_indices)
        pcs = self.get_critical_connections(cth=cth, s=self.max_iter)['pcs']
        path = self.path_data
        for pc in pcs:
            x = path[:, pc]
            dt = np.median(np.diff(np.sort(t)))  # Or choose your desired time step
            t_reg = np.arange(t.min(), t.max(), dt)
            interp_func = interp1d(t, x, kind='linear', fill_value='extrapolate')
            x_reg = interp_func(t_reg)
            # forward_autocorr = correlate(x_reg - x_reg.mean(), x_reg - x_reg.mean(), mode='full')
            forward_autocorr = correlate(x_reg, x_reg, mode='full')
            # autocorr = autocorr / np.max(autocorr)
            lags = np.arange(-t.max() + dt, t.max(), dt)
            idx = lags > 0
            ax1.plot(lags[idx] / 100000, forward_autocorr[idx], lw=1, color=color_list[2], alpha=0.5)
            x_reg_back = x_reg[::-1]
            # backward_autocorr = correlate(x_reg - x_reg.mean(), x_reg_back - x_reg_back.mean(), mode='full')
            backward_autocorr = correlate(x_reg, x_reg_back, mode='full')
            ir = np.mean((forward_autocorr - backward_autocorr) ** 2)  # mean square divergence between forward and backward
            irs.append(ir)
        vmax = np.max(irs)
        ax2.hist(irs, bins=50, range=(0, vmax), color=color_list[2], density=True, alpha=0.5)
        indices = np.random.randint(0, 784 * 50 + 50 + 50 * 50 + 50 + 50 * 10 + 10, size=(1000,))
        irs = []
        for i in indices:
            if i not in pcs:
                t = np.array(self.test_point_indices)
                x = path[:, i]
                dt = np.median(np.diff(np.sort(t)))  # Or choose your desired time step
                t_reg = np.arange(t.min(), t.max(), dt)
                interp_func = interp1d(t, x, kind='linear', fill_value='extrapolate')
                x_reg = interp_func(t_reg)
                # forward_autocorr = correlate(x_reg - x_reg.mean(), x_reg - x_reg.mean(), mode='full')
                forward_autocorr = correlate(x_reg, x_reg, mode='full')
                # autocorr = autocorr / np.max(autocorr)
                lags = np.arange(-t.max() + dt, t.max(), dt)
                idx = lags > 0
                ax1.plot(lags[idx] / 100000, forward_autocorr[idx], lw=1, color='black', alpha=0.5)
                x_reg_back = x_reg[::-1]
                # backward_autocorr = correlate(x_reg - x_reg.mean(), x_reg_back - x_reg_back.mean(), mode='full')
                backward_autocorr = correlate(x_reg, x_reg_back, mode='full')
                ir = np.mean((forward_autocorr - backward_autocorr) ** 2)
                irs.append(ir)
        ax2.hist(irs, bins=50, range=(0, vmax), color='black', density=True, alpha=0.5)
        ax1.set_xlabel(r"$\tau$")
        ax1.set_xticks([0, 0.5, 1, 1.5, 2])
        ax1.set_ylabel(r"$R$")  # Autocorrelation
        ax1.spines[['right', 'top']].set_visible(False)
        ax1.text(2.2, -32, r"$\times 10^5$", fontsize=16)
        ax2.set_yscale('log')
        plt.savefig(self.save_folder + "trajectories_irreversible.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Trajectory irreversibility")

    def per_data_landscape(self, show_iters=(1, 100, 200000)):
        cmap = plt.get_cmap("seismic")
        names = ["cc", "cn", "nn"]
        N = self.data.shape[0]
        for s in show_iters:
            data = self.get_landscape(s=s)
            vmax = 0
            vmin = 1e10
            idx = np.random.randint(0, N, 30)
            for ii in range(3):
                ml = data[f"{names[ii]}_L"][:, :, idx]
                vmax = max(vmax, ml.max())
                vmin = min(vmin, ml.min())
            for ii in range(3):
                X = data[f"{names[ii]}_X"]
                Y = data[f"{names[ii]}_Y"]
                l = data[f"{names[ii]}_L"]
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')

                for ind in idx[ii * 10:ii * 10 + 10]:
                    ax.plot_surface(X, Y, l[:, :, ind], cmap=cmap, vmin=vmin, vmax=vmax)
                remove_background(ax)
                plt.savefig(self.save_folder + f"per_data_landscape_{names[ii]}_iter{s}.svg",
                            dpi=600, bbox_inches="tight", transparent=True)
                plt.close()

            set_figure(3)
            fig, ax = plt.subplots(figsize=(1, 3))
            ax.set_visible(False)
            norm = Normalize(vmin=vmin, vmax=vmax)
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # for older matplotlib versions
            cbar = plt.colorbar(sm, ax=ax, shrink=0.3)
            cbar.set_label('Loss')
            plt.savefig(self.save_folder + f"per_data_landscape_cbar_iter{s}.svg",
                        dpi=600, bbox_inches="tight", transparent=True)
            plt.close()
        self.info("Per-data landscape")

    def effective_landscape(self, show_iters=(1, 100, 200000)):
        cmap = plt.get_cmap("seismic")
        names = ["cc", "cn", "nn"]
        for si, s in enumerate(show_iters):
            data = self.get_landscape(s=s)
            vmax = 0
            vmin = 1e10
            for ii in range(3):
                ml = np.mean(data[f"{names[ii]}_L"], axis=2)
                vmax = max(vmax, ml.max())
                vmin = min(vmin, ml.min())
            for ii in range(3):
                X = data[f"{names[ii]}_X"]
                Y = data[f"{names[ii]}_Y"]
                ll = data[f"{names[ii]}_rand_L"]
                ml = np.mean(data[f"{names[ii]}_L"], axis=2)

                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, ll, cmap=cmap, vmin=vmin, vmax=vmax)
                remove_background(ax)
                plt.savefig(self.save_folder + f"effect_landscape_{names[ii]}_iter{s}.svg",
                            dpi=600, bbox_inches="tight", transparent=True)
                plt.close()

                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, ml, cmap=cmap, vmin=vmin, vmax=vmax)
                remove_background(ax)
                plt.savefig(self.save_folder + f"mean_landscape_{names[ii]}_iter{s}.svg",
                            dpi=600, bbox_inches="tight", transparent=True)
                plt.close()
            set_figure(3)
            fig, ax = plt.subplots(figsize=(1, 3))
            ax.set_visible(False)
            norm = Normalize(vmin=vmin, vmax=vmax)
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # for older matplotlib versions
            cbar = plt.colorbar(sm, ax=ax, shrink=0.3)
            cbar.set_label('Loss')
            plt.savefig(self.save_folder + f"landscape_cbar_iter{s}.svg",
                        dpi=600, bbox_inches="tight", transparent=True)
            plt.close()
        self.info("Effective and mean landscape")

    def conc_ineq(self):
        set_figure(4)
        data = self.get_conc_ineq()
        fig, ax = plt.subplots()
        b = data['b']
        ss = self.test_point_indices
        norm = LogNorm(vmin=1, vmax=ss[-1])
        cmap = self.iter_cmap
        for si, s in enumerate(ss):
            pb = data[f"iter{s}"]
            color = cmap(norm(s + 1))
            plt.plot(b, pb, lw=1, color=color)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$\ln t$')
        cbar.set_ticks([])
        # plt.legend(title="Iteration", frameon=False, loc=(1, 0.2))
        plt.xlabel(r"$b$")
        plt.ylabel(r"$P(L \geq b)$")
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xticks([0, 1, 2, 3])
        plt.savefig(self.save_folder + "concentration_inequality.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("concentration_inequality")

    def per_data_loss_dist(self, show_iters=(0, 1, 5, 10, 100, 200000)):
        set_figure(4)
        loss_dict = self.get_per_data_loss()
        ss = show_iters
        norm = LogNorm(vmin=1, vmax=ss[-1])
        cmap = plt.get_cmap('jet')
        fig, ax = plt.subplots()
        for si, s in enumerate(ss):
            p, e = hist(loss_dict[f"iter{s}"])
            color = cmap(norm(s + 1))
            ax.plot(e, p, lw=1, label=f"{s}", color=color)
        # ax.legend(title=r"$s$", frameon=False, loc=(1, 0.4), ncol=1)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # for older matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$\ln t$')
        cbar.set_ticks([])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r"$L$")
        ax.set_ylabel(r"$p(L)$")
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + "data_wise_loss.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("data_wise_loss")

    def irreversibility_dist(self):
        irs = self.get_irreversibility()
        p, e = hist(irs, bins=100)

        def power_law(x, a, b):
            return a * x + b

        set_figure(4)
        fig, ax = plt.subplots()
        idx = e > e[0]
        x = e[idx]
        y = p[idx]
        ax.plot(x, y, color=color_list[5], ls='none', marker='.', ms=10, alpha=0.5)
        idx = (e > e[0]) & (e < 100)
        xx = e[idx]
        yy = p[idx]
        popt, pcov = curve_fit(power_law, np.log(xx), np.log(yy))
        x = np.linspace(x[0], x.max(), 100)
        ax.plot(x, np.exp(power_law(np.log(x), *popt)), ls="-", color='black', lw=2,
                label=rf"$\alpha = {-popt[0]:.2f}$")
        ax.legend(frameon=False, loc="lower left", fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r"$R$")
        ax.set_ylabel(r"$p(R)$")
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(f"./figures/init0_lr0.1_bs64_neurons50/irreversibility_dist.svg", dpi=600,
                    bbox_inches="tight", transparent=True)
        plt.close()
        self.info("irreversibility_dist")

    def redundant_conn(self, trans_iter=100, end_iter=200000):
        s1, s2 = trans_iter, end_iter
        spks1 = self.get_spks(s=s1)['spks']
        spks2 = self.get_spks(s=s2)['spks']
        corr1 = np_pearson_corr(spks1, spks1)
        corr2 = np_pearson_corr(spks2, spks2)
        cc = corr1 - corr2
        set_figure(3)
        fig, axs = plt.subplots(1, 3, figsize=(7, 7 / 3))
        axs[0].imshow(np.abs(corr1) > 0.5)
        axs[1].imshow(np.abs(corr2) > 0.5)
        axs[2].imshow(np.abs(cc) > 0.5)
        axs[0].set_title(r"$t = 100$")
        axs[1].set_title(r"$t = 2\times 10^5$")
        axs[2].set_title("Difference")
        axs[0].axis('off')
        axs[1].axis('off')
        axs[2].axis('off')
        plt.savefig(self.save_folder + "corr_difference.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        self.info("Difference of critical connections")

        th = 0.5
        ix, iy = np.where(np.abs(cc) > th)
        n = self.hyperparam[3]
        neuron_pos = [(0, i) for i in range(n)] + [(1, i) for i in range(n)] + [(2, i) for i in range(10)]
        params = [p.detach().cpu().numpy() for p in self.mlp.parameters() if p.requires_grad]
        pp = [np.zeros_like(x) for x in params]
        for x, y in zip(ix, iy):
            npx, npy = neuron_pos[x], neuron_pos[y]
            if npx[0] != npy[0]:
                if npx[0] > npy[0]:
                    npx, npy = npy, npx
                if npx[0] == 0:
                    pp[2][npy[1], npx[1]] = 1
                if npx[0] == 1:
                    pp[4][npy[1], npx[1]] = 1
            else:
                if npx[0] == 0:
                    for i in range(50):
                        npz = (1, i)
                        vx = corr2[neuron_pos.index(npx), neuron_pos.index(npz)]
                        vy = corr2[neuron_pos.index(npy), neuron_pos.index(npz)]
                        if np.abs(vx) > th and np.abs(vy) > th:
                            pp[2][npz[1], npx[1]] = 1
                            pp[2][npz[1], npy[1]] = 1
                if npx[0] == 1:
                    for i in range(10):
                        npz = (2, i)
                        vx = corr2[neuron_pos.index(npx), neuron_pos.index(npz)]
                        vy = corr2[neuron_pos.index(npy), neuron_pos.index(npz)]
                        if np.abs(vx) > th and np.abs(vy) > th:
                            pp[4][npz[1], npx[1]] = 1
                            pp[4][npz[1], npy[1]] = 1
                if npx[0] == 2:
                    for i in range(50):
                        npz = (1, i)
                        vx = corr2[neuron_pos.index(npx), neuron_pos.index(npz)]
                        vy = corr2[neuron_pos.index(npy), neuron_pos.index(npz)]
                        if np.abs(vx) > th and np.abs(vy) > th:
                            pp[4][npx[1], npz[1]] = 1
                            pp[4][npy[1], npz[1]] = 1

        pp = np.concatenate([x.reshape(-1) for x in pp], axis=0)
        pcs = np.where(pp > 0)[0]
        set_figure(4)
        fig = plt.figure()
        points = []
        ax1 = fig.add_axes((0.1, 0.1, 0.7, 0.8))
        ax2 = fig.add_axes((0.82, 0.1, 0.18, 0.8))
        test_point_indices = self.test_point_indices
        path = self.path_data
        for pc in pcs:
            ax1.plot(test_point_indices, path[:, pc], lw=0.5, color=color_list[1], alpha=0.5)
            points.append(path[-1, pc])
        ax2.hist(points, bins=20, orientation='horizontal', color=color_list[1], alpha=0.5, density=True)
        indices = np.random.randint(0, 784 * 50 + 50 + 50 * 50 + 50 + 50 * 10 + 10, size=(1000,))
        points = []
        for i in indices:
            if i not in pcs:
                ax1.plot(test_point_indices, path[:, i], lw=0.5, color='black', alpha=0.5)
                points.append(path[-1, i])
        ax2.hist(points, bins=20, orientation='horizontal', color='black', alpha=0.5, density=True)
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_yticks([])
        ax2.set_xticks([5])
        ax1.set_xscale('log')
        ax1.set_xlabel(r"$t$")
        ax1.set_ylabel(r"$w$")
        ax1.set_xlim([1, 200000])
        ax1.set_xticks([1, 1000, 100000])
        plt.savefig(self.save_folder + "trajectories_corr_decay.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Trajectory for redundant connections")

        set_figure(4)
        fig = plt.figure()
        ax1 = fig.add_axes((0.1, 0.1, 0.9, 0.9))
        ax2 = fig.add_axes((0.8, 0.5, 0.47, 0.47))
        irs = []
        t = np.array(test_point_indices)
        for pc in pcs:
            x = path[:, pc]
            dt = np.median(np.diff(np.sort(t)))  # Or choose your desired time step
            t_reg = np.arange(t.min(), t.max(), dt)
            interp_func = interp1d(t, x, kind='linear', fill_value='extrapolate')
            x_reg = interp_func(t_reg)
            # forward_autocorr = correlate(x_reg - x_reg.mean(), x_reg - x_reg.mean(), mode='full')
            forward_autocorr = correlate(x_reg, x_reg, mode='full')
            # autocorr = autocorr / np.max(autocorr)
            lags = np.arange(-t.max() + dt, t.max(), dt)
            idx = lags > 0
            ax1.plot(lags[idx] / 100000, forward_autocorr[idx], lw=1, color=color_list[2], alpha=0.5)
            x_reg_back = x_reg[::-1]
            # backward_autocorr = correlate(x_reg - x_reg.mean(), x_reg_back - x_reg_back.mean(), mode='full')
            backward_autocorr = correlate(x_reg, x_reg_back, mode='full')
            ir = np.mean((forward_autocorr - backward_autocorr) ** 2)  # mean square divergence between forward and backward
            irs.append(ir)
        vmax = np.max(irs)
        ax2.hist(irs, bins=50, range=(0, vmax), color=color_list[2], density=True, alpha=0.5)
        indices = np.random.randint(0, 784 * 50 + 50 + 50 * 50 + 50 + 50 * 10 + 10, size=(1000,))
        irs = []
        for i in indices:
            if i not in pcs:
                t = np.array(test_point_indices)
                x = path[:, i]
                dt = np.median(np.diff(np.sort(t)))  # Or choose your desired time step
                t_reg = np.arange(t.min(), t.max(), dt)
                interp_func = interp1d(t, x, kind='linear', fill_value='extrapolate')
                x_reg = interp_func(t_reg)
                # forward_autocorr = correlate(x_reg - x_reg.mean(), x_reg - x_reg.mean(), mode='full')
                forward_autocorr = correlate(x_reg, x_reg, mode='full')
                # autocorr = autocorr / np.max(autocorr)
                lags = np.arange(-t.max() + dt, t.max(), dt)
                idx = lags > 0
                ax1.plot(lags[idx] / 100000, forward_autocorr[idx], lw=1, color='black', alpha=0.5)
                x_reg_back = x_reg[::-1]
                # backward_autocorr = correlate(x_reg - x_reg.mean(), x_reg_back - x_reg_back.mean(), mode='full')
                backward_autocorr = correlate(x_reg, x_reg_back, mode='full')
                ir = np.mean((forward_autocorr - backward_autocorr) ** 2)
                irs.append(ir)
        ax2.hist(irs, bins=50, range=(0, vmax), color='black', density=True, alpha=0.5)
        ax1.set_xlabel(r"$\tau$")
        ax1.set_xticks([0, 0.5, 1, 1.5, 2])
        ax1.set_ylabel(r"$R$")  # Autocorrelation
        ax1.spines[['right', 'top']].set_visible(False)
        ax1.text(2.2, -32, r"$\times 10^5$", fontsize=16)
        ax2.set_yscale('log')
        plt.savefig(self.save_folder + "trajectories_irreversible.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Trajectory irreversibility for redundant connections")

        set_figure(4)
        fig = plt.figure()
        ax1 = fig.add_axes((0.1, 0.1, 0.9, 0.9))
        ax2 = fig.add_axes((0.8, 0.5, 0.47, 0.47))
        irs = []
        for pc in pcs:
            t = np.array(test_point_indices)
            x = path[:, pc]
            dt = np.median(np.diff(np.sort(t)))  # Or choose your desired time step
            t_reg = np.arange(t.min(), t.max(), dt)
            interp_func = interp1d(t, x, kind='linear', fill_value='extrapolate')
            x_reg = interp_func(t_reg)
            forward_autocorr = correlate(x_reg, x_reg, mode='full')
            lags = np.arange(-t.max() + dt, t.max(), dt)
            idx = lags > 0
            ax1.plot(lags[idx] / 100000, forward_autocorr[idx], lw=1, color=color_list[2], alpha=0.5)
            x_reg_back = x_reg[::-1]
            backward_autocorr = correlate(x_reg, x_reg_back, mode='full')
            ir = np.mean((forward_autocorr - backward_autocorr) ** 2)  # mean square divergence between forward and backward
            irs.append(ir)
        vmax = np.max(irs)
        ax2.hist(irs, bins=50, range=(0, vmax), color=color_list[2], density=True, alpha=0.5)
        indices = np.random.randint(0, 784 * 50 + 50 + 50 * 50 + 50 + 50 * 10 + 10, size=(1000,))
        irs = []
        for i in indices:
            if i not in pcs:
                t = np.array(test_point_indices)
                x = path[:, i]
                dt = np.median(np.diff(np.sort(t)))  # Or choose your desired time step
                t_reg = np.arange(t.min(), t.max(), dt)
                interp_func = interp1d(t, x, kind='linear', fill_value='extrapolate')
                x_reg = interp_func(t_reg)
                forward_autocorr = correlate(x_reg, x_reg, mode='full')
                # autocorr = autocorr / np.max(autocorr)
                lags = np.arange(-t.max() + dt, t.max(), dt)
                idx = lags > 0
                ax1.plot(lags[idx] / 100000, forward_autocorr[idx], lw=1, color='black', alpha=0.5)
                x_reg_back = x_reg[::-1]
                backward_autocorr = correlate(x_reg, x_reg_back, mode='full')
                ir = np.mean((forward_autocorr - backward_autocorr) ** 2)
                irs.append(ir)
        ax2.hist(irs, bins=50, range=(0, vmax), color='black', density=True, alpha=0.5)
        ax2.set_xlim([0, 350])
        ax1.set_xlabel("Lags", fontsize=16)
        ax1.set_xticks([0, 0.5, 1, 1.5, 2])
        ax1.set_ylabel(r"$R$")
        ax1.spines[['right', 'top']].set_visible(False)
        ax2.set_yscale('log')
        ax1.text(2.2, -32, r"$\times 10^5$", fontsize=16)
        plt.savefig(self.save_folder + "trajectories_irreversible_decay.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Trajectory irreversibility for decay")

        self.update_model(s=trans_iter)
        model1 = copy.deepcopy(self.model)
        self.update_model(s=end_iter)
        model2 = copy.deepcopy(self.model)
        params = self.params
        flat_params1 = flatten_params(model1.parameters())
        flat_params2 = flatten_params(model2.parameters())
        x, y = self.data, self.target
        unflattened_params = unflatten_params(flat_params2, params)
        y_hat = x.reshape(x.shape[0], -1) @ unflattened_params[0].T + unflattened_params[1]
        y_hat = F.relu(y_hat)
        y_hat = y_hat @ unflattened_params[2].T + unflattened_params[3]
        y_hat = F.relu(y_hat)
        y_hat = y_hat @ unflattened_params[4].T + unflattened_params[5]
        losses = self.criterion(y_hat, y).detach().cpu().numpy()
        prs = {}
        prs[f"no_redundant"] = []
        b = np.linspace(0, 3, 100)
        n_data = self.data.shape[0]
        for bi in b:
            pr = np.sum(losses >= bi) / n_data
            prs[f"no_redundant"].append(pr)
        prs["mean_loss_no_redundant"] = np.mean(losses)

        for ind in pcs:
            flat_params2[ind] = flat_params1[ind]
        unflattened_params = unflatten_params(flat_params2, params)
        y_hat = x.reshape(x.shape[0], -1) @ unflattened_params[0].T + unflattened_params[1]
        y_hat = F.relu(y_hat)
        y_hat = y_hat @ unflattened_params[2].T + unflattened_params[3]
        y_hat = F.relu(y_hat)
        y_hat = y_hat @ unflattened_params[4].T + unflattened_params[5]
        losses = self.criterion(y_hat, y).detach().cpu().numpy()
        prs[f"with_redundant"] = []
        for bi in b:
            pr = np.sum(losses >= bi) / n_data
            prs[f"with_redundant"].append(pr)
        prs["mean_loss_with_redundant"] = np.mean(losses)

        set_figure(4)
        fig, ax = plt.subplots()
        ax.plot(b, prs["no_redundant"], lw=1, color=color_list[0], label="No redundant")
        ax.plot(b, prs["with_redundant"], lw=1, color=color_list[1], label="With redundant")
        ax1 = fig.add_axes([0.6, 0.3, 0.35, 0.35])
        x = ["No redundant", "With redundant"]
        h = [prs["mean_loss_no_redundant"], prs["mean_loss_with_redundant"]]
        ax1.bar(0, h[0], width=0.1, color=color_list[0])
        ax1.bar(0.3, h[1], width=0.1, color=color_list[1])
        ax1.set_xlim([-0.2, 0.5])
        ax1.set_xticks([])
        ax1.set_ylabel(r"$\langle L \rangle$", fontsize=16)
        ax1.set_yscale('log')
        ax.legend(frameon=False, loc=(0.2, 0.8), fontsize=18)
        ax.set_xlabel(r"$b$")
        ax.set_ylabel(r"$P(L \geq b)$")
        ax.set_yscale('log')
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + "concentration_inequality_compare.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Concentration inequality compare for redundant connections")

    def loss_mean_var(self, show_iters=(1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000)):
        set_figure(4)
        loss_dict = self.get_per_data_loss()
        fig, ax = plt.subplots()
        ss = show_iters
        locs = [str(s) for s in ss]
        for si, s in enumerate(ss):
            pos = si + 1
            if si == 1:
                ax.bar(s - 0.1 * s, np.mean(loss_dict[f"iter{s}"]), width=0.2 * s, color=color_list[4], label=r"$\langle L \rangle$")
                ax.bar(s + 0.1 * s, np.var(loss_dict[f"iter{s}"]), width=0.2 * s, color=color_list[5], label=r"$Var(L)$")
            else:
                ax.bar(s - 0.1 * s, np.mean(loss_dict[f"iter{s}"]), width=0.2 * s, color=color_list[4])
                ax.bar(s + 0.1 * s, np.var(loss_dict[f"iter{s}"]), width=0.2 * s, color=color_list[5])
        ax.legend(frameon=False, loc=(0.7, 0.8), fontsize=18)
        ax.set_xticks(np.array(ss), locs)
        ax.set_xlabel(r"$t$")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + "change_of_per_data_loss_mean_var.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Change of per-data loss mean and variance")

    def grad_mean_var(self, show_iters=(0, 1, 5, 10, 100, 200000)):
        ss = show_iters
        pcs_dict = self.get_pcs(show_iters)
        set_figure(3)
        fig, axs = plt.subplots(2, 3, figsize=(3.5 * 3, 3.5 * 2))
        for si, s in enumerate(ss):
            data = self.get_grad_cov_parallel(s=s, save_gm_mean_var=True)
            gm_mean = data['gm_mean']
            gm_var = data['gm_var']
            mask = np.ones_like(gm_mean)
            pcs = pcs_dict[f"iter{s}"]
            mask[pcs] = 0
            mask = mask > 0
            if si < 3:
                ax = axs[0, si]
            else:
                ax = axs[1, si - 3]
            m1, = ax.plot(gm_mean[mask] ** 2, gm_var[mask], ls='none', label="none-critical",
                          marker='.', ms=1, alpha=0.5, color='black')
            m2, = ax.plot(gm_mean[pcs] ** 2, gm_var[pcs], ls='none', label="critical",
                          marker='o', markerfacecolor='none', ms=6, alpha=0.5, color='red')
            if s < 1000:
                for a in np.linspace(0, 50, 100):
                    r = np.sum((gm_var / gm_mean ** 2) < a) / gm_mean.shape[0]
                    if r > 0.01:
                        break
            else:
                for a in np.linspace(19000, 21000, 100):
                    r = np.sum((gm_var / gm_mean ** 2) < a) / gm_mean.shape[0]
                    if r > 0.01:
                        break
            x = np.linspace(np.min(gm_mean ** 2), np.max(gm_mean ** 2), 100)
            l1, = ax.plot(x, a * x, lw=1, color='green', label=f"slope: {a:.2f}")
            # if si == 0:
            #     legend1 = plt.legend(handles=[m1, m2], bbox_to_anchor=(1.5, 1.2), frameon=False, ncol=1)
            #     plt.gca().add_artist(legend1)  # Add the first legend manually
            ax.legend(handles=[l1], loc="upper left", frameon=False)

            if si in [0, 3]:
                ax.set_ylabel(r"$Var(L)$")
            if si in [3, 4, 5]:
                ax.set_xlabel(r"$\langle L \rangle^2$")
        plt.savefig(self.save_folder + "var_g_mean_g.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Gradient mean and variance")

        set_figure(3)
        fig, axs = plt.subplots(2, 3, figsize=(3.5 * 3, 3.5 * 2))
        for si, s in enumerate(ss):
            data = self.get_grad_cov_parallel(s=s, save_gm_mean_var=True)
            gm_mean = data['gm_mean']
            gm_var = data['gm_var']
            mask = np.ones_like(gm_mean)
            pcs = pcs_dict[f"iter{s}"]
            mask[pcs] = 0
            mask = mask > 0
            if si < 3:
                ax = axs[0, si]
            else:
                ax = axs[1, si - 3]

            p, e = hist(gm_mean[pcs] ** 2, bins=50)
            ax.plot(e, p, lw=1, label="mean critical", color='red')
            p, e = hist(gm_mean[mask] ** 2, bins=50)
            ax.plot(e, p, lw=1, label="mean non-critical", color='black')
            p, e = hist(gm_var[pcs], bins=50)
            ax.plot(e, p, lw=1, label="var critical", color='red', ls='--')
            p, e = hist(gm_var[mask], bins=50)
            ax.plot(e, p, lw=1, label="var non-critical", color='black', ls='--')
            if si == 0:
                ax.legend(loc=(0.1, 1), frameon=False, ncol=4)

            if si == 0 or si == 3:
                ax.set_ylabel("Probability density")
            if si in [3, 4, 5]:
                ax.set_xlabel("Value")
            ax.set_xscale('log')
            ax.set_yscale('log')
        plt.savefig(self.save_folder + "var_g_mean_g_dist.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Gradient mean and variance distribution (critical connections)")

    def neural_corr_graph(self, cth=0.5, show_iters=(0, 1, 5, 10, 100, 200000)):
        ss = show_iters
        for si, s in enumerate(ss):
            fig, ax = plt.subplots()
            spks = self.get_spks(s=s)['spks']
            corr = np_pearson_corr(spks, spks)
            G = nx.Graph()
            edge_colors = []
            # for i in range(corr.shape[0]):
            #     G.add_node(f'{i}', color='black')
            for i in range(corr.shape[0]):
                for j in range(corr.shape[1]):
                    if i == j:
                        continue
                    if np.abs(corr[i, j]) > cth:
                        if corr[i, j] > 0:
                            G.add_edge(f'{i}', f'{j}', value=corr[i, j], color=color_list[4])
                        else:
                            G.add_edge(f'{i}', f'{j}', value=-corr[i, j], color=color_list[5])
            cmap = plt.get_cmap("plasma")
            pos = nx.spring_layout(G, seed=13648)
            nx.draw(G, ax=ax, node_size=15, pos=pos, node_color='black',
                    edge_color=[color for color in nx.get_edge_attributes(G, 'color').values()],
                    width=[(w - 0.48) * 10 for w in nx.get_edge_attributes(G, 'value').values()],
                    edge_cmap=cmap)
            plt.axis('on')
            plt.savefig(self.save_folder + f"neural_corr_graph_iter{s}.svg",
                        dpi=600, bbox_inches="tight", transparent=True)
            plt.close()
        self.info("Neural correlation graph")

    def diff_pc(self, trans_iter=100, end_iter=200000):
        test_point_indices = self.test_point_indices
        path = self.path_data
        pcs_dict = self.get_pcs()
        pcs1 = pcs_dict[f"iter{trans_iter}"]
        pcs2 = pcs_dict[f"iter{end_iter}"]
        data = self.get_grad_cov_parallel(s=trans_iter, save_gm_mean_var=True)
        gm_mean1 = data['gm_mean']
        gm_var1 = data['gm_var']
        gm_mean2 = data['gm_mean']
        gm_var2 = data['gm_var']
        path_index = np.where(np.array(test_point_indices) == trans_iter)[0]
        param1 = path[path_index.item()]
        path_index = np.where(np.array(test_point_indices) == end_iter)[0]
        param2 = path[path_index.item()]
        diff_pcs = []
        for pc in pcs1:
            if pc not in pcs2:
                diff_pcs.append(pc)
        set_figure(3)
        fig, ax = plt.subplots()
        mean_g = gm_mean1[diff_pcs]
        var_g = gm_var1[diff_pcs]
        weight = param1[diff_pcs]
        p, e = hist(weight * (-mean_g), bins=10)
        plt.plot(e, p, lw=1, color=color_list[0], label="Mean")
        v = np.where(p == p.max())[0]
        plt.vlines(e[v], ymin=-0.05, ymax=0.7, color=color_list[0], ls='--')
        mean_g = gm_mean1[pcs2]
        var_g = gm_var1[pcs2]
        weight = param1[pcs2]
        p, e = hist(weight * (-mean_g), bins=10)
        plt.plot(e, p, lw=1, color=color_list[1], label="Variance")
        v = np.where(p == p.max())[0]
        ymin, ymax = plt.ylim()
        plt.vlines(e[v], ymin=-0.05, ymax=0.7, color=color_list[1], ls='--')
        plt.xlabel("Gradient value")
        plt.ylabel("Probability density")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.legend(frameon=False, loc=(0.7, 0.8))
        plt.tight_layout()
        plt.savefig(self.save_folder + f"diff_pcs_weight_grad_{trans_iter}_{end_iter}.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()

        set_figure(3)
        fig, ax = plt.subplots()
        mean_g = gm_mean1[diff_pcs]
        var_g = gm_var1[diff_pcs]
        weight = param1[diff_pcs]
        pos = np.sum(weight * (-mean_g) > 0) / len(diff_pcs)
        ax.bar(1 - 0.1, pos, width=0.2, color=color_list[4], label=f"Redundant")
        neg = np.sum(weight * (-mean_g) < 0) / len(diff_pcs)
        ax.bar(2 - 0.1, neg, width=0.2, color=color_list[4])
        mean_g = gm_mean1[pcs2]
        var_g = gm_var1[pcs2]
        weight = param1[pcs2]
        pos = np.sum(weight * (-mean_g) > 0) / len(pcs2)
        ax.bar(1 + 0.1, pos, width=0.2, color=color_list[5], label="Critical")
        neg = np.sum(weight * (-mean_g) < 0) / len(pcs2)
        ax.bar(2 + 0.1, neg, width=0.2, color=color_list[5])
        ax.legend(loc=(0.8, 0.8), frameon=False)
        ax.set_xticks((1, 2), ["Align", "Diverge"])
        ax.set_ylabel("Connection ratio")
        ax.set_xlim([0, 3])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        plt.savefig(self.save_folder + f"diff_pcs_weight_grad_bar_{trans_iter}_{end_iter}.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()

        set_figure(3)
        fig, ax = plt.subplots()
        mean_g = gm_mean1[diff_pcs]
        var_g = gm_var1[diff_pcs]
        weight = param1[diff_pcs]
        p, e = hist(mean_g, bins=10)
        plt.plot(e, p, lw=1, color=color_list[4], label="Redundant")
        p, e = hist(var_g, bins=10)
        plt.plot(e, p, lw=1, color=color_list[4], ls='--')
        mean_g = gm_mean1[pcs2]
        var_g = gm_var1[pcs2]
        weight = param1[pcs2]
        p, e = hist(mean_g, bins=10)
        plt.plot(e, p, lw=1, color=color_list[5], label="Critical")
        p, e = hist(var_g, bins=10)
        plt.plot(e, p, lw=1, color=color_list[5], ls='--')
        plt.legend(loc="upper right", frameon=False)
        plt.xlabel("Gradient value")
        plt.ylabel("Probability density")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        plt.savefig(self.save_folder + f"diff_pcs_weight_grad_dist_compare_{trans_iter}_{end_iter}.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Difference of critical connections")

    def param_dist_evolution(self, show_iters=(0, 5, 10, 100, 500, 10000, 100000, 200000)):
        assert len(show_iters) == 8, "Exactly 8 figures is needed."
        set_figure(4)
        ss = show_iters
        test_point_indices = self.test_point_indices
        path = self.path_data
        fig, axs = plt.subplots(2, 4, figsize=(3.5 * 4, 3.5 * 2))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        names = [r'$t = 0$', r'$t = 5$', r'$t = 10$', r'$t = 100$', r'$t = 500$',
                 r'$t = 1 \times 10^4$', r'$t = 1 \times 10^5$', r'$t = 2 \times 10^5$']
        for i, s in enumerate(ss):
            path_index = np.where(np.array(test_point_indices) == s)[0]
            params = path[path_index.item()]
            if i < 4:
                ax = axs[0, i]
            else:
                ax = axs[1, i - 4]
            p, e = hist(params[params >= 0], bins=100)
            ax.plot(e, p, color=color_list[0], lw=1)
            p, e = hist(-params[params <= 0], bins=100)
            ax.plot(e, p, color=color_list[1], lw=1)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_title(names[i])
            if i in [4, 5, 6, 7]:
                ax.set_xlabel("Strength")
            if i == 0 or i == 4:
                ax.set_ylabel("Probability density")
        plt.savefig(self.save_folder + "param_dist_evolution.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Weight distribution evolution")

    def fit_param_dist(self, s=0):
        set_figure(4)
        fig, ax = plt.subplots()

        def power_law(x, a, b):
            return a * x ** b

        path = self.path_data
        path_index = np.where(np.array(self.test_point_indices) == s)[0]
        params = path[path_index.item()].copy()
        p, e = hist(params[params >= 0], bins=100)
        m1, = plt.plot(e, p, color=color_list[0], lw=1, label="Positive", ls='none', marker='.', ms=10)
        p, e = hist(-params[params < 0], bins=100)
        m2, = plt.plot(e, p, color=color_list[1], lw=1, label="Negative", ls='none', marker='.', ms=10)
        idx = e > 0.15
        popt, pcov = curve_fit(power_law, e[idx], p[idx])
        idx = e > 0.08
        fit_x, fit_y = e[idx], power_law(e[idx], *popt)
        # randomize connectivity
        for _ in range(int(1e5)):
            i1, i2 = np.random.randint(0, params.shape[0], size=(2,))
            d = np.random.randn() / 10
            params[i1] += d
            params[i2] -= d
        p, e = hist(params[params >= 0], bins=100)
        m3, = plt.plot(e, p, color=color_list[2], lw=1, label="Randomized", ls='none', marker='.', ms=10)
        m4, = plt.plot(fit_x, fit_y, color='black', lw=3, label=rf"$\alpha = {-popt[1]:.2f}$")
        plt.xscale('log')
        plt.yscale('log')
        # legend1 = plt.legend(handles=[m4], loc='upper left', frameon=False)
        # plt.gca().add_artist(legend1)  # Add the first legend manually
        plt.legend(frameon=False, loc='lower left', fontsize=16)
        ax.spines[['right', 'top']].set_visible(False)
        plt.xlabel(r"w")
        plt.ylabel(r"$p(w)$")
        plt.savefig(self.save_folder + "param_fit.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("param_fit")

    def entropy_large_dev(self, batch_size=64, show_iters=(10, 50, 100, 500, 1000, 5000, 10000, 100000, 200000)):
        set_figure(1)
        fig, axs = plt.subplots(3, 3)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        ss = show_iters
        data = self.get_per_data_loss()
        NL = 100000
        B = batch_size
        for si, s in enumerate(ss):
            ax = axs[si // 3, si % 3]
            lk = data[f"iter{s}"]
            l_hat = []
            for i in range(NL):
                idx = np.random.randint(0, lk.shape[0], size=(B,))
                l_hat.append(np.mean(lk[idx]))
            ths = np.linspace(np.mean(l_hat), np.max(l_hat), 100)
            temp = np.zeros_like(ths)
            entropy = np.zeros_like(ths)
            p_entropy = np.zeros_like(ths)
            p_l_hat = np.zeros_like(ths)
            for bi, b in enumerate(ths):
                temp[bi] = get_temperature(lk, b)
                entropy = quasi_entropy(lk, temp[bi])
                dS = entropy[bi] - np.log(lk.shape[0])
                p_l_hat[bi] = np.sum(l_hat >= b) / NL
                p_entropy[bi] = np.exp(B * dS)
            idx = p_l_hat > 0
            ax.plot(ths[idx], p_l_hat[idx], ls='-', lw=1, color='black')
            ax.plot(ths[idx], p_entropy[idx], ls=":", lw=1, color='red')
            ax.set_yscale('log')
            ax.set_title(f"Iteration {s}")
        plt.savefig(f"./figures/init0_lr0.1_bs64_neurons50/entropy_large_dev.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Entropy large deviation")

    def entropy_large_dev_bs(self,
                             s=100,
                             batch_sizes=(2, 8, 32, 64, 256, 1024, 2048, 4096, 16384),
                             show_iters=(10, 50, 100, 500, 1000, 5000, 10000, 100000, 200000)):
        set_figure(1)
        fig, axs = plt.subplots(3, 3)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        Bs = batch_sizes
        ss = show_iters
        data = self.get_per_data_loss()
        NL = 100000
        for si, B in enumerate(Bs):
            ax = axs[si // 3, si % 3]
            lk = data[f"iter{s}"]
            l_hat = []
            for i in range(NL):
                idx = np.random.randint(0, lk.shape[0], size=(B,))
                l_hat.append(np.mean(lk[idx]))
            ths = np.linspace(np.mean(l_hat), np.max(l_hat), 100)
            temp = np.zeros_like(ths)
            entropy = np.zeros_like(ths)
            p_entropy = np.zeros_like(ths)
            p_l_hat = np.zeros_like(ths)
            for bi, b in enumerate(ths):
                temp[bi] = get_temperature(lk, b)
                entropy[bi] = quasi_entropy(lk, temp[bi])
                dS = entropy[bi] - np.log(lk.shape[0])
                p_l_hat[bi] = np.sum(l_hat >= b) / NL
                p_entropy[bi] = np.exp(B * dS)
            idx = p_l_hat > 0
            ax.plot(ths[idx], p_l_hat[idx], ls='-', lw=1, color='black')
            ax.plot(ths[idx], p_entropy[idx], ls=":", lw=1, color='red')
            ax.set_yscale('log')
            ax.set_title(f"Batch size {B}")
        plt.savefig(f"./figures/init0_lr0.1_bs64_neurons50/entropy_large_dev_bs.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Entropy large deviation for batch size effect")

    def entropy_beta_fix_w(self, s=100):
        set_figure(3)
        fig, ax = plt.subplots()
        data = self.get_per_data_loss()
        lk = data[f"iter100"]
        Bs = np.linspace(np.mean(lk), np.max(lk), 100)
        entropy = []
        temp = []
        for b in Bs:
            t = get_temperature(lk, b)
            e = quasi_entropy(lk, t)
            entropy.append(e)
            temp.append(t)
        ax.plot(Bs, entropy, lw=1, color=color_list[0], label=r'$S$')
        ax.plot(Bs[:-1], temp[:-1], lw=1, color=color_list[1], label=r'$\beta$')
        ax.legend(fontsize=12, frameon=False)
        ax.set_xlabel(r"$b$")
        plt.savefig(f"./figures/init0_lr0.1_bs64_neurons50/entropy_beta_fix_w.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Beta with boundary")

    def lagrange(self, bth=5e-3):
        set_figure(3)
        fig, ax = plt.subplots()
        data = self.get_dS_energy(bth=bth)
        entropy = data['dS']
        energy = data['energy']
        r = data['lambda']
        lagrangian = data['lagrangian']
        ss = self.test_point_indices
        ax1 = ax.twinx()
        ax1.plot(ss, np.log(entropy), lw=1, color=color_list[0], label=r'$\ln \Delta S$')
        ax.plot(0, 0, lw=1, color=color_list[0], label=r'$\ln \Delta S$')
        # ax.plot(test_point_indices, np.log(r*entropy), lw=1, color=color_list[0], label=r'$\ln \Delta S$')
        ax.plot(ss, energy, lw=1, color=color_list[1], label=r'$\bar{L}$')
        # plt.plot(test_point_indices, temp, lw=1, label=r'$\beta$')
        ax.plot(ss, lagrangian + 0.5, lw=1, color=color_list[2],
                label=r'$\mathcal{L}=\bar{L}-\lambda \Delta S$')
        ax.legend(frameon=False, fontsize=12, title_fontsize=12, loc=(0.2, 0.8))
        ax.set_xlabel(r"t")
        ax.set_xscale('log')
        # plt.yscale('log')
        ax.spines[['right', 'top']].set_visible(False)
        ax1.spines[['top']].set_visible(False)
        ax.spines['left'].set_color(color_list[1])
        ax1.spines['right'].set_color(color_list[0])
        ax.tick_params(axis='y', colors=color_list[1])
        ax1.tick_params(axis='y', colors=color_list[0])
        print(r)
        plt.savefig(self.save_folder + f"lagrange_b{bth}.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Lagrange")

    def driven_forces(self, bth=5e-3):
        set_figure(3)
        data = self.get_dS_energy(bth=bth)
        entropy = data['dS']
        energy = data['energy']
        r = data['lambda']
        fig, ax = plt.subplots()
        dS = np.diff(r * entropy)
        dE = np.diff(energy)
        ss = self.test_point_indices
        x = np.array(ss[:-1])
        # y = np.abs(dS) - np.abs(dE)
        # idx = y >= 0
        # ax.plot(x[idx], y[idx], lw=1, ls='none', marker='.', ms=2,
        #         color=color_list[0], label="Entropy driven")
        # idx = y < 0
        # ax.plot(x[idx], -y[idx], lw=1, ls='none', marker='.', ms=2,
        #         color=color_list[1], label="Energy driven")
        ax.plot(x, dS, lw=1, color=color_list[0], label=r"$\lambda \nabla \Delta S$")
        ax.plot(x, dE, lw=1, color=color_list[1], label=r"$\nabla \bar{L}$")
        ax.set_xlabel(r"$t$")
        ax.set_xscale('log')
        ax.legend(frameon=False, fontsize=12, loc=(0.4, 0.8))
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + "driven_force_b{bth}.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Driven forces")

    def entropy_change_th(self, show_iters=(1, 10, 50, 100, 10000, 20000, 30000, 40000)):
        set_figure(3)
        fig, ax = plt.subplots()
        ss = show_iters
        edata = self.get_entropies(ss)
        N = np.log(self.data.shape[0])
        sths = np.linspace(1, np.log(N), 5)
        for sti, sth in enumerate(sths):
            shift = []
            for si, s in enumerate(ss):
                entropy = edata[f"step{s}_entropy"]
                dS = np.log(N) - entropy
                b = edata[f"step{s}_Bs"]
                spline = UnivariateSpline(dS, b, k=3, s=0)
                shift.append(spline(sth))
            ax.plot(ss, shift, lw=1, color=color_list[sti], label=f"{sth:.4f}")
        ax.set_xscale('log')
        ax.legend(frameon=False, fontsize=12, title=r"$\Delta S_{th}$", loc=(1.0, 0.2))
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$b$")
        # ax.vlines(x=40000, ymin=0, ymax=17)
        plt.savefig(self.save_folder + f"entropy_change_th.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Entropy change threshold")

    def entropy_b_all_w(self):
        set_figure(3)
        cmap = self.iter_cmap
        fig, ax = plt.subplots()
        N = self.data.shape[0]
        ss = self.test_point_indices
        norm = LogNorm(vmin=1, vmax=ss[-1])
        edata = self.get_entropies()
        for si, s in enumerate(ss):
            entropy = edata[f"step{s}_entropy"]
            dS = np.log(N) - entropy
            b = edata[f"step{s}_Bs"]
            b = b - b.min()
            color = cmap(norm(s + 1))
            plt.plot(b, dS, lw=1, color=color)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(r'$t$')
        cbar.set_ticks([1, 100, 10000])
        ax.set_xlabel(r"$b$")
        ax.set_ylabel(r"$\Delta S$")
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(f"./figures/init0_lr0.1_bs64_neurons50/entropy_beta_all_w.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Entropy for all weights")

    def test_spline_fit(self, s=10000):
        set_figure(3)
        data = self.get_per_data_loss()
        lk = data[f'iter{s}']
        es = []
        N = self.data.shape[0]
        Bs = np.linspace(np.mean(lk), np.max(lk), 10)
        for b in Bs:
            t = get_temperature(lk, b)
            e = quasi_entropy(lk, t)
            es.append(e)
        es = np.array(es)
        dS = np.log(N) - es
        b = Bs - Bs.min()
        fig, ax = plt.subplots()
        plt.plot(b, dS, lw=1, color=color_list[0], label="Empirical")
        x = np.linspace(0, 10, 100)
        spline = UnivariateSpline(b, dS, k=5, s=0)
        ax.plot(x, spline(x), lw=1, color='k', label="Spline fit")
        ax.legend(fontsize=12, frameon=False)
        ax.set_xlabel(r"$b$")
        ax.set_ylabel(r"$\Delta S$")
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + f"entropy_spline_fit.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Spline fit")

    def ccg_umap(self):
        set_figure(3)
        data = self.get_ccg_umap_cluster()
        fig, ax = plt.subplots()
        embedding = data["embedding"]
        u_nodes = data["u_nodes"]
        c_nodes = data["cluster_center"]
        cmap = self.iter_cmap
        ax.scatter(*embedding.T, s=100, c=u_nodes, cmap=cmap)
        ax.plot(*c_nodes.T, lw=2)
        ax.scatter(*embedding.T, s=50, c=u_nodes, cmap=cmap)
        ax.plot(*c_nodes.T, lw=2)
        plt.savefig(self.save_folder + "ccg_umap_embedding.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("CCG umap clustering")

    def get_neurons_edges(
            self,
            label="umap",  # "clique", "critical neurons", "critical edges"
            data_loader="train",  # "test"
    ):
        if label == "umap":
            cluster_id = self.get_ccg_umap_cluster()["cluster_id"]
            data = self.cluster_id_to_weight_dict(cluster_id)
        elif label == "clique":
            cluster_id = self.get_ccg_clique_cluster()["cluster_id"]
            data = self.cluster_id_to_weight_dict(cluster_id)
        elif label == "critical neurons":
            data = self.get_ccg_critical_neurons()
        elif label == "critical edges":
            data = self.get_ccg_critical_edges()
        else:
            raise NotImplementedError("Only available: umap, clique, critical neurons, critical edges.")
        return data

    def ccg_loss_acc(
            self,
            label="umap",  # "clique", "prune"
            data_loader="train",  # "test"
    ):
        set_figure(2)
        data = self.get_neurons_edges(label, data_loader)
        perform_data = self.get_ccg_performance(data, label=label, data_loader=data_loader)

        reserve_neurons = ("All", "Critical", "Non-critical")
        reserve_keys = ["all", "critical", "non-critical"]
        x = np.arange(len(reserve_neurons))  # the label locations
        width = 0.075  # the width of the bars
        fig, ax = plt.subplots()
        for name, value in perform_data.items():
            if "loss" in name:
                items = name.split("_")
                ti = int(items[0].strip("task"))
                ki = reserve_keys.index(items[2])
                offset = width * ti
                if "all" in name:
                    rects = ax.bar(ki + offset, np.mean(value), width, label=f"{ti}", color=color_list[ti])
                else:
                    rects = ax.bar(ki + offset, np.mean(value), width, color=color_list[ti])
                # ax.bar_label(rects, padding=3)
        if label == "critical edges":
            ax.set_xlabel("Reserved connections")
        else:
            ax.set_xlabel("Reserved neurons")
        ax.set_ylabel(r"$\bar{L}$")
        ax.set_xticks(x + 4.5 * width, reserve_neurons)
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left", ncols=1, frameon=False)
        plt.savefig(self.save_folder + f"ccg_loss_{label}_{data_loader}.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()

        width = 0.075  # the width of the bars
        fig, ax = plt.subplots()
        for name, value in perform_data.items():
            if "acc" in name:
                items = name.split("_")
                ti = int(items[0].strip("task"))
                ki = reserve_keys.index(items[2])
                offset = width * ti
                if "all" in name:
                    rects = ax.bar(ki + offset, value, width, label=f"{ti}", color=color_list[ti])
                else:
                    rects = ax.bar(ki + offset, value, width, color=color_list[ti])
                # ax.bar_label(rects, padding=3)
        if label == "critical edges":
            ax.set_xlabel("Reserved connections")
        else:
            ax.set_xlabel("Reserved neurons")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(x + 4.5 * width, reserve_neurons)
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left", ncols=1, frameon=False)
        ax.axhline(y=1.0, lw=1, ls='--', color='black')
        plt.savefig(self.save_folder + f"ccg_acc_{label}_{data_loader}.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("CCG loss accuracy")

    def ccg_loss_dist(
            self,
            label="umap",  # "clique", "prune"
            data_loader="train",  # "test"
    ):
        set_figure(3)
        data = self.get_neurons_edges(label, data_loader)
        perform_data = self.get_ccg_performance(data, label=label, data_loader=data_loader)

        fig, ax = plt.subplots()
        for name, value in perform_data.items():
            if "reserve_critical_loss" in name:
                items = name.split("_")
                ti = int(items[0].strip("task"))
                p, e = hist(value, bins=100)
                ax.plot(e, p, lw=1, color=color_list[ti], label=f"{ti}")
        ax.legend(loc=(1, 0), fontsize=12, frameon=False)
        ax.set_xscale("log")
        ax.set_yscale('log')
        ax.set_xlabel(r"$L_k$")
        ax.set_ylabel(r"$p(L_k)$")
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + f"ccg_loss_dist_{label}_{data_loader}.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("CCG loss dist")

    def ccg_graph(
            self,
            label="umap",
            data_loader="train",  # "test"
    ):
        data = self.get_neurons_edges(label, data_loader)
        graphs, sorts = self.weight_dict_to_graph(data)

        set_figure(10)
        init, lr, bs, neurons = self.hyperparam
        fig, axs = plt.subplots(1, 10, figsize=(4.5 * 10, 3 * 1))
        for i, graph in enumerate(graphs):
            ax = axs[i]
            ax.set_title(f"{i}")
            ax.scatter(np.ones((neurons,)) * 0, np.arange(neurons), s=5, c='black')
            ax.scatter(np.ones((neurons,)) * 1, np.arange(neurons), s=5, c='black')
            # ax.scatter(np.ones((10,)) * 2, np.arange(10), s=5, c='black')
            ax.scatter(np.ones((10,)) * 2, np.linspace(0, neurons, 10), s=5, c='black')
            nx.draw_networkx_nodes(graph, nx.get_node_attributes(graph, 'coordinates'),
                                   node_size=20,
                                   node_color=[color for color in nx.get_node_attributes(graph, 'color').values()],
                                   ax=ax)
            nx.draw_networkx_edges(graph, nx.get_node_attributes(graph, 'coordinates'),
                                   edge_color=[color for color in nx.get_edge_attributes(graph, 'color').values()],
                                   ax=ax)
            ax.axis('off')
        plt.savefig(self.save_folder + f"ccg_graph_{label}_{data_loader}.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("CCG graphs")

    def ccg_rastermap(
            self,
            label="umap",  # "clique", "prune"
            data_loader="train",  # "test"
            show_iters=(0, 1, 5, 10, 100, 200000),
    ):
        data = self.get_neurons_edges(label, data_loader)
        graphs, sorts = self.weight_dict_to_graph(data)

        set_figure(3)
        ss = show_iters
        for s in ss:
            data = self.get_spks(s=s)
            spks = data['spks']
            num_data = data['num_data']
            nbin = 1  # number of neurons to bin over
            sn = utils.bin1d(spks[sorts], bin_size=nbin, axis=0)  # bin over neuron axis
            # only plot the first 50 images
            start_idx = np.array(num_data).cumsum()[:-1]
            idx = np.concatenate([np.arange(200)] + [np.arange(200) + i for i in start_idx], axis=0)
            # plot_data = np.concatenate([data, sn[:, idx]], axis=0)
            plot_data = sn[:, idx]
            fig, ax = plt.subplots(1, 1, figsize=(3, 2))
            plt.imshow(plot_data, cmap="gray_r", vmin=0, vmax=0.8, aspect="auto")
            # for i in range(10):
            #     plt.text(x=i * 256 + 120, y=-20, s=f"{i}", fontdict={'size': 36})
            ax.set_axis_off()
            plt.savefig(self.save_folder + f"ccg_rastermap_iter{s}_{label}_{data_loader}.svg",
                        dpi=600, bbox_inches="tight", transparent=True)
            plt.close()
        self.info("CCG Rastermap")

    def ccg_correlation_matrix(
            self,
            label="umap",  # "clique", "prune"
            data_loader="train",  # "test"
            show_iters=(0, 1, 5, 10, 100, 200000),
    ):
        data = self.get_neurons_edges(label, data_loader)
        graphs, sorts = self.weight_dict_to_graph(data)

        set_figure(3)
        ss = show_iters
        cmap = plt.get_cmap('seismic')
        for s in ss:
            spks = self.get_spks(s=s)['spks']
            fig, ax = plt.subplots(1, 1, figsize=(3, 2))
            corr = np_pearson_corr(spks[sorts], spks[sorts])
            cax = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
            ax.axis("off")
            plt.savefig(self.save_folder + f"ccg_corr_iter{s}_{label}_{data_loader}.svg",
                        dpi=600, bbox_inches="tight", transparent=True)
            plt.close()
        self.info("Correlation matrix")

    def loss_trajectories(self, num=1000):
        trajectories = self.get_loss_trajectory()
        set_figure(3)
        fig, ax = plt.subplots()
        N = self.data.shape[0]
        idx = np.random.randint(0, N, size=(num,))
        ss = self.test_point_indices
        for i in idx:
            ax.plot(ss, trajectories[i, :], lw=0.1, color='k', alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$L_k$")
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + "loss_trajectory.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Loss trajectories")

    def loss_flux(self, bins=100):
        flux_data = self.get_loss_flux(bins)
        set_figure(3)
        fig, ax = plt.subplots()
        flux = flux_data["flux"]
        ss = self.test_point_indices
        total_flux = np.zeros((len(ss[:-1]),))
        for ti in range(len(ss[:-1])):
            total_flux[ti] = np.sum(np.abs(flux[:, ti]))
        ax.plot(ss[:-1], total_flux, lw=1, color='k')
        ax1 = fig.add_axes([0.6, 0.4, 0.5, 0.5])
        ax1.plot(ss[:-1], total_flux, lw=1, color='k')
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$|\hat{J}(t)|$")
        ax.set_xticks([100, 100000])
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + "loss_flux.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Loss flux")

    def weight_trajectories(self, num=1000):
        set_figure(3)
        trajectories = self.path_data.T
        N = trajectories.shape[0]
        ss = self.test_point_indices
        set_figure(3)
        fig, ax = plt.subplots()
        idx = np.random.randint(0, N, size=(num,))
        for i in idx:
            ax.plot(ss, trajectories[i, :], lw=0.1, color='k', alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$w$")
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + "weight_trajectory.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Weight trajectories")

    def weight_flux(self, bins=100):
        set_figure(3)
        fig, ax = plt.subplots()
        flux_data = self.get_weight_flux(bins)
        flux = flux_data["flux"]
        ss = self.test_point_indices
        total_flux = np.zeros((len(ss[:-1]),))
        for ti in range(len(ss[:-1])):
            total_flux[ti] = np.sum(np.abs(flux[:, ti]))
        ax.plot(ss[:-1], total_flux, lw=1, color='k')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$|\hat{J}(t)|$")
        ax.set_xticks([100, 100000])
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + "weight_flux.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Weight flux")

    def ccg_activity_coherence(self, t=0, s=0):
        spks = self.get_spks(s=s)['spks']
        self.update_model(s=s)
        set_figure(2)
        fig = plt.figure()
        ax1 = fig.add_axes([0.1, 0.82, 1, 0.18], xticklabels=[])
        ax2 = fig.add_axes([0.1, 0.1, 1, 0.7])
        l = 0
        es = self.get_ccg_critical_edges()[f"task{t}_edges"]
        idx = np.lexsort((es[:, 1], es[:, 0]))
        es = es[idx]
        used_neurons = []
        for e in es:
            li, pi, lj, pj = e
            i = self.neuron_pos.index((li, pi))
            j = self.neuron_pos.index((lj, pj))
            if i in used_neurons:
                continue
            else:
                used_neurons.append(i)
            wij = self.weight_from_neurons(i, j)
            wsi = wij * spks[i]
            ss, th = scale_to_range(wsi, (l, l + 1))
            ax2.plot(ss, ls='none', marker='.', ms=1, alpha=0.1, color=color_list[(l+1)%10])
            ax2.hlines(y=th, xmin=0, xmax=60000, color='black', ls='--', lw=1)
            ax2.plot(6e4 + 3e3, l + 0.5, ls='none', marker='.', ms=10, color=color_list[(l+1)%10])
            ax2.text(6e4 + 5e3, l + 0.3, f"N{i}")
            l += 1
        sj = spks[j]
        ss, th = scale_to_range(sj, (l, l + 3))
        ax1.plot(ss, ls='none', marker='.', ms=1, alpha=0.1, color=color_list[0])
        ax1.hlines(y=th, xmin=0, xmax=60000, color='black', ls='--', lw=1)
        ax1.plot(6e4 + 3e3, l + 1.5, ls='none', marker='.', ms=10, color=color_list[0])
        ax1.text(6e4 + 5e3, l + 1.2, f"N{j}")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_yticks([])
        ax2.set_xlim([-3e3, 7.5e4])
        ax1.set_xlim([-3e3, 7.5e4])
        ax2.set_xlabel(r"$x$")
        # ax2.legend(bbox_to_anchor=(1, 1), loc="upper left", ncols=1, frameon=False, fontsize=16)
        ax2.set_ylabel(r"$\hat{a}_i(x)$")
        plt.savefig(self.save_folder + f"ccg_activity_coherence_task{t}_iter{s}.jpg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("CCG activity coherence")

    def activity_overlap(self, s=0):
        set_figure(3)
        fig, ax = plt.subplots()
        spks = self.get_spks(s=s)['raw']
        num_data = self.get_spks(s=s)['num_data']
        num_data = np.cumsum(num_data)
        num_data = np.concatenate([np.array([0]), num_data], axis=0)
        for i in range(self.num_tasks):
            si = spks[100 + i]
            smin, smax = np.min(si), np.max(si)
            a, b = num_data[i], num_data[i + 1]
            p, e = hist(si[a:b], bins=50, bd=(smin, smax))
            plt.plot(e, p, lw=1, color=color_list[i], label=f"{i}")
            sj = np.concatenate([si[:a], si[b:]], axis=0)
            p, e = hist(sj, bins=50, bd=(smin, smax))
            ax.plot(e, p, lw=1, color='black')
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left", ncols=1, frameon=False, fontsize=12)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylabel(r"$p(a_{\mathcal{T}})$")
        ax.set_xlabel(r"$a_{\mathcal{T}}$")
        plt.savefig(self.save_folder + f"activity_overlap_iter{s}.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("CCG activity overlap")

    def ccg_coupling_energy_spectrum(self, s=0, t=0):
        set_figure(3)
        fig, ax = plt.subplots()
        es = self.get_ccg_critical_edges()[f"task{t}_edges"]
        self.update_model(s=s)
        spks = self.get_spks(s=s)['spks']
        num_data = self.get_spks(s=s)['num_data']
        num_data = np.cumsum(num_data)
        num_data = np.concatenate([np.array([0]), num_data], axis=0)
        E = 0
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
            E = E + wsi
        for ti in range(10):
            a, b = num_data[ti], num_data[ti + 1]
            plt.plot(np.arange(b - a) + num_data[ti], E[a:b], ls='none', marker='.', ms=1, alpha=0.2, color=color_list[ti])
        ax.set_yscale('log')
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$-\mathcal{E}_\mathcal{G}(x)$")
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + f"ccg_coupling_energy_spectrum_iter{s}_task{t}.jpg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("CCG coupling energy spectrum")

    def ccg_coupling_energy(self, s=0, t=0):
        set_figure(3)
        fig, ax = plt.subplots()
        es = self.get_ccg_critical_edges()[f"task{t}_edges"]
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
        for ti in range(10):
            a, b = num_data[ti], num_data[ti + 1]
            E = np.mean(ss[a:b])
            ax.bar(ti, E, color=color_list[ti])
        ax.set_xlabel(r"$\mathcal{T}$")
        ax.set_ylabel(r"$-E_\mathcal{G}(\mathcal{T})$")
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + f"ccg_coupling_energy_iter{s}_task{t}.jpg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("CCG coupling energy")

    def ccg_coupling_energy_evolve(self):
        set_figure(3)
        fig, ax = plt.subplots()
        data = self.get_ccg_coupling_energy_evolve()
        for t in range(self.num_tasks):
            E = data[f"task{t}"]
            ax.plot(self.test_point_indices, E, lw=1, color=color_list[t])
        ax.set_xscale('log')
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$E_\mathcal{G}(t)$")
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + f"ccg_coupling_energy_evolve.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("CCG coupling energy evolve")

    def connection_property_evolution_sim(self, p_Hebb=0.37):
        data_dict = np.load(self.save_folder + f"hebb_sim_data_empirical_corr_p{p_Hebb}.npz")
        N = 110
        E = int(N * (N - 1) / 2)
        I, J = np.triu_indices(N, k=1)
        edge_inds = np.ravel_multi_index((I, J), (N, N))
        steps = []
        de = []
        fn = []
        cc = []
        hetero = []
        CW = []
        for key, val in data_dict.items():
            if 'W' in key:
                steps.append(int(key[1:]))
                de.append(density(val, 0))
                cc.append(clustering_coefficient(val, 0))
                hetero.append(heterogeneity(val, I, J, 0))
                # print(key, np.any(val<0))
            if 'C' in key:
                fn.append(np.sum(val[I, J] ** 2))
        ts = self.test_point_indices
        for t in ts[1:]:
            W = data_dict[f"W{int(t)}"]
            C = data_dict[f"C{int(t)}"]
            W = W / np.sum(W)
            C = C / np.sum(C)
            CW.append(np.mean(W - C))

        de = np.array(de)
        fn = np.array(fn)
        cc = np.array(cc)
        hetero = np.array(hetero)
        set_figure(4)
        fig, axs = plt.subplots()
        # plt.plot(steps, de, lw=1, label="Density")
        plt.plot(steps, fn / fn.max(), lw=1, label="Frobenius norm", color=color_list[0])
        plt.plot(steps, cc, lw=1, label="Clustering coefficient", color=color_list[1])
        hetero = np.nan_to_num(hetero)
        plt.plot(steps, hetero, lw=1, label="Heterogeneity", color=color_list[2])
        plt.plot(steps, CW, lw=1, label=r"Mean probability divergence", color=color_list[4])
        plt.legend(frameon=False, loc=(0.1, 0.7), fontsize=12)
        # plt.xscale('log')
        # plt.yscale('log')

        plt.xlabel(r"$t$")
        axs.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.save_folder + f"hebb_sim_empirical_corr_property_evolution_p{p_Hebb}.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Connection weight property evolution")

    def connection_dist_fit_sim(self, p_Hebb=0.37):
        data = np.load(self.save_folder + f"hebb_sim_empirical_corr_dist_data_p{p_Hebb}.npz")
        Ws = data['Ws']
        set_figure(4)
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        # p, e = hist(Ws[Ws>=0].reshape(-1), bins=1000)
        p, e = hist(np.abs(Ws).reshape(-1), bins=100)
        e = e / 100
        plt.plot(e, p, ls='none', marker='.', ms=10, lw=1, color=color_list[3], label="Positive")
        # x_new = np.linspace(e[1], e.max(), 100)
        # spline = make_smoothing_spline(np.log(e), np.log(p), lam=0.5)  # k=3 for cubic
        # y_smooth = spline(np.log(x_new))
        # x_new = np.concatenate([e[0].reshape(-1), x_new], axis=0)
        # y_smooth = np.concatenate([np.log(p[0]).reshape(-1), y_smooth], axis=0)
        # plt.plot(x_new, np.exp(y_smooth), lw=1, color='black', ls=":")

        idx = e >= e[1]
        # idx = e >= e[0]
        popt, pcov = curve_fit(power_law, np.log(e[idx]), np.log(p[idx]))
        x = np.linspace(e[0], e.max(), 100)
        plt.plot(x, np.exp(power_law(np.log(x), *popt)), ls="-", color='black', lw=3, label=rf"$\alpha = {popt[0]:.2f}$")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(frameon=False, loc="lower left", fontsize=15)
        ax.spines[['right', 'top']].set_visible(False)
        plt.xlabel(r"$w$")
        plt.ylabel(r"$p(w)$")
        plt.savefig(self.save_folder + f"hebb_sim_empirical_corr_conn_dist_fit_p{p_Hebb}.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Connection distribution fit for simulation")

    def exponent_p_Hebb_relation(self, pps=np.linspace(0.1, 0.9, 50)):
        exponents = []
        for pp in pps:
            data = np.load(self.save_folder + f"hebb_sim_empirical_corr_dist_data_p{pp}.npz")
            Ws = data['Ws']
            p, e = hist(np.abs(Ws).reshape(-1), bins=100)
            e = e / 100
            idx = e >= e[1]
            popt, pcov = curve_fit(power_law, np.log(e[idx]), np.log(p[idx]))
            alpha = popt[0]
            exponents.append(alpha)
        exponents = np.array(exponents)
        set_figure(4)
        fig, ax = plt.subplots()
        ax.plot(1 / pps, -exponents, ls='none', marker=".", ms=5, color=color_list[0], alpha=0.5, label="Simulation")

        def linear(x, a, b):
            return a * x + b

        x = 1 / pps
        y = -exponents
        idx = x < 6
        x = x[idx]
        y = y[idx]
        popt, pcov = curve_fit(linear, x, y)
        x = np.linspace(0, 6, 100)
        y = linear(x, *popt)
        ax.plot(x, y, lw=2, color='black', label=f"Slope: {popt[0]:.2f}")
        ax.legend(frameon=True, loc="lower right", fontsize=15)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_xticks([0, 5, 10])
        # ax.text(x=3, y=2, s=f"~{popt[0]:.2f}")
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlabel(r"$1/p_C$")
        ax.set_ylabel(r"$\alpha$")
        plt.savefig(self.save_folder + "hebb_sim_exponents_p_Hebb_relation.svg",
                    dpi=600, bbox_inches="tight", transparent=True)
        plt.close()
        self.info("Exponent and p_Hebb relation for simulation")


