from utils import *
from glob import glob
from visualizer import NNVisualizer

init = 0
lr = 0.1
bs = 64
neurons = 50
max_iter = 200000
test_point_indices = [i for i in range(0, 50)] + [i for i in range(50, 500, 10)] + [i for i in range(500, max_iter, 500)] + [max_iter]
nnv = NNVisualizer(max_iter=max_iter, test_point_indices=test_point_indices, hyperparam=(init, lr, bs, neurons), num_workers=0,
                   data_folder="./autodl-tmp/", save_folder="./figures/")
# for s in tqdm(test_point_indices):
#     # print(f"neurons {neurons}, Testing point {s}", flush=True)
#     nnv.get_activity_key_measures(s=s, treat_negatives='abs', use_fisher_z=False, keep_activity_files=True)

# nnv.get_activity_key_measures(s=max_iter, treat_negatives='abs', use_fisher_z=False, keep_activity_files=True)
# nnv.NCG_property_evolution_model_size()
# nnv.NCG_property_evolution_batch_size()
# nnv.NCG_property_evolution_learning_rate()
# nnv.NCG_property_evolution_mobility_factor()
# nnv.NCG_degree_component_clique_dist(aim="degree")
# nnv.NCG_degree_component_clique_dist(aim="component")
# nnv.NCG_degree_component_clique_dist(aim="clique")
# nnv.NCG_degree_component_clique_dist(aim="clique_value")
# nnv.NCG_degree_dist_compare(iter=-1)
# nnv.NCG_degree_dist_compare(iter=100)
# nnv.NCG_clique_dist_compare(iter=-1)
# nnv.NCG_clique_dist_compare(iter=100)
# nnv.NCG_clique_value_compare()
# nnv.NCG_property_model_size()
# nnv.NCG_property_model_size_final_iter()
# nnv.NCG_network_properties()

# nnv.rastermap()
# nnv.correlation_matrix()
# nnv.cumulative_explained_variance()
# nnv.cumulative_explained_variance_PC()
# nnv.activity_power_law()
# nnv.activity_power_law_zoom()
# nnv.activity_sparsity()
# nnv.corr_dist()
# nnv.cross_corr()
# nnv.activity_measures()
# nnv.loss_acc()
# nnv.clique_dist()
# nnv.pred_dist()
# nnv.pred_entropy()
# nnv.complexity_pred_entropy()
# nnv.sparsity_pred_entropy()
# nnv.raw_activities()
# nnv.num_NCG_edges_minibatch()
# nnv.clique_dist_fit()
# nnv.clique_size_cth()
# nnv.clique_size_iter()
# nnv.num_NCG_edges()
# nnv.variance_corr_hebbin()
# nnv.trajectory_autocorr()
# nnv.trajectory_irreversibility()
# nnv.per_data_landscape()
# nnv.effective_landscape()
# nnv.conc_ineq()
# nnv.per_data_loss_dist()
# nnv.redundant_conn()
# nnv.loss_mean_var()
# nnv.grad_mean_var()
# nnv.neural_corr_graph()
# nnv.diff_pc()
# nnv.param_dist_evolution()
# nnv.fit_param_dist(s=max_iter)
# nnv.entropy_large_dev()
# nnv.entropy_large_dev_bs()
# nnv.entropy_beta_fix_w()
# nnv.lagrange()
# nnv.driven_forces()
# nnv.entropy_b_all_w()
# nnv.test_spline_fit()

# nnv.ccg_umap()
# nnv.ccg_graph(label="umap", data_loader="train")
# nnv.ccg_rastermap(label="umap", data_loader="train")
# nnv.ccg_correlation_matrix(label="umap", data_loader="train")
# nnv.ccg_loss_acc(label="umap", data_loader="train")

# nnv.ccg_graph(label="clique", data_loader="train")
# nnv.ccg_rastermap(label="clique", data_loader="train")
# nnv.ccg_correlation_matrix(label="clique", data_loader="train")
# nnv.ccg_loss_acc(label="clique", data_loader="train")

# nnv.ccg_graph(label="critical neurons", data_loader="train")
# nnv.ccg_rastermap(label="critical neurons", data_loader="train")
# nnv.ccg_correlation_matrix(label="critical neurons", data_loader="train")
# nnv.ccg_loss_acc(label="critical neurons", data_loader="train")

# nnv.ccg_graph(label="critical edges", data_loader="train")
# nnv.ccg_rastermap(label="critical edges", data_loader="train")
# nnv.ccg_correlation_matrix(label="critical edges", data_loader="train")
# nnv.ccg_loss_acc(label="critical edges", data_loader="train")
# nnv.ccg_loss_acc(label="critical edges", data_loader="test")
# data = nnv.get_neurons_edges(label="critical edges", data_loader="train")
# nnv.get_ccg_performance_all_task(data, label="critical edges", data_loader="train")
# nnv.ccg_loss_dist(label="critical edges", data_loader="train")
# for redundant connections
##### corr1 = nnv.get_corr_mat(s=max_iter)['corr']
##### print(np.sum(corr1))
##### corr2 = nnv.get_corr_mat(s=100)['corr']
##### print(np.sum(corr2))
##### corr1 = corr1 > 0.8
##### corr2 = corr2 > 0.5
##### diff = corr2.astype(int) - corr1.astype(int)
##### ix, iy = np.where(diff > 0)
##### i = ix[1]
##### j = iy[1]
##### print(i, j)
##### nnv.correlation_energy_spectrum(100, i, j)
##### nnv.correlation_energy(100, i, j)
corr = nnv.get_corr_mat(s=max_iter)['corr']
corr = corr - np.diag(np.ones((corr.shape[0],)))
ix, iy = np.where(corr > 0.8)
i, j = ix[0], iy[0]
print(i, j)
for s in [1, 10, 100, 200000]:
    # nnv.activity_overlap(s=s)
    for t in range(10):
        nnv.ccg_coupling_energy_spectrum(s=s, t=t)
        nnv.ccg_coupling_energy(s=s, t=t)
        # nnv.ccg_activity_coherence(t=t, s=s)

