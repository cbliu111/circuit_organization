import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numba
import torch
import scipy
from scipy import optimize
from scipy import linalg, sparse, stats
import networkx as nx
import ruptures as rpt

from scipy.optimize import curve_fit
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from tqdm import tqdm
import copy
from scipy.interpolate import UnivariateSpline


@numba.njit(parallel=True)
def fast_knn_indices_from_precomputed(X, n_neighbors):
    knn_indices = np.empty((X.shape[0], n_neighbors), dtype=np.int32)
    for row in numba.prange(X.shape[0]):
        # v = np.argsort(X[row])  # Need to call argsort this way for numba
        v = X[row].argsort(kind="quicksort")
        v = v[:n_neighbors]
        knn_indices[row] = v
    return knn_indices


# ---------- Preprocessing ----------
def fisher_z_transform(R):
    R = np.clip(R, -0.999999, 0.999999)
    Z = 0.5 * np.log((1 + R) / (1 - R))
    np.fill_diagonal(Z, 0.0)
    return Z


def symmetrize(M):
    return 0.5 * (M + M.T)


def zero_diagonal(M):
    M = M.copy()
    np.fill_diagonal(M, 0.0)
    return M


def keep_nonnegative(M):
    # If using correlation matrices, you may (optionally) zero out negatives
    # depending on your analysis choice.
    M = M.copy()
    M[M < 0] = 0.0
    return M


# ---------- Graph construction ----------
def proportional_threshold(W, density):
    # Keep top density fraction of absolute weights
    # return binary network
    N = W.shape[0]
    triu_idx = np.triu_indices(N, k=1)
    weights = np.abs(W[triu_idx])
    k = int(np.floor(density * len(weights)))
    if k <= 0:
        A = np.zeros_like(W)
        return A
    thr = np.partition(weights, -k)[-k]
    A = np.zeros_like(W)
    sel = (W >= thr)
    A[sel] = 1.0
    A = np.triu(A, 1)
    A = A + A.T
    np.fill_diagonal(A, 0.0)
    return A


def weighted_threshold(W, density):
    # Return weighted adjacency after sparsification by density
    A = proportional_threshold(W, density)
    Wth = W * A
    return Wth


def hard_threshold(W, th):
    A = np.zeros_like(W)
    sel = (W >= th)
    A[sel] = 1.0
    A = np.triu(A, 1)
    A = A + A.T
    np.fill_diagonal(A, 0.0)
    return A


# ---------- Metrics: components / percolation ----------
def largest_component_fraction(A):
    """
    the fraction of nodes in the largest connected component,
    often called the “giant component fraction” or “order parameter” in percolation theory.
    Subcritical regime (below criticality): When the average degree ⟨k⟩ is small (for ER graphs, ⟨k⟩ = p(n − 1) < 1),
    the graph consists of many small tree-like components.
    The largest component size scales like O(log n), so the fraction max_comp/n → 0 as n grows.
    Critical point: Around ⟨k⟩ ≈ 1 for ER graphs, the system undergoes a continuous (second-order) phase transition.
    Component size distribution becomes broad, with a largest component scaling like O(n^(2/3)) at criticality.
    The fraction still goes to 0 as n → ∞, but finite-size networks show a noticeable bump and strong fluctuations.
    Supercritical regime (above criticality): For ⟨k⟩ > 1, a “giant component” emerges containing a positive fraction of nodes.
    The output of the function increases rapidly from near 0 to a sizable fraction (e.g., 0.2–0.9 depending on parameters).
    As more edges are added (or fewer are removed), this fraction grows and eventually can approach 1 if nearly all nodes become connected.
    Finite-size effects smooth the transition; the “sharpness” increases with larger n.

    """
    G = nx.from_numpy_array(A)
    if A.sum() == 0:
        return 0.0
    comp_sizes = [len(c) for c in nx.connected_components(G)]
    return max(comp_sizes) / A.shape[0]


def component_size_stats(A):
    G = nx.from_numpy_array(A)
    comps = [len(c) for c in nx.connected_components(G)]
    if len(comps) == 0:
        return 0.0, 0.0
    s = np.array(comps, dtype=float)
    s_excl_gcc = s[s < s.max()]
    if len(s_excl_gcc) == 0:
        return s.max(), 0.0
    susceptibility = (np.mean(s_excl_gcc ** 2) - np.mean(s_excl_gcc) ** 2) / (np.mean(s_excl_gcc) + 1e-12)
    return s.max(), susceptibility


# ---------- Metrics: paths and efficiencies ----------
def global_efficiency(W, weighted=True):
    # W: adjacency matrix; if weighted=True, treat weights as strengths, convert to distances = 1/weight
    N = W.shape[0]
    if weighted:
        # Avoid division by zero; for zero edges, distance = inf
        with np.errstate(divide='ignore', invalid='ignore'):
            D = np.where(W > 0, 1.0 / W, np.inf)
            G = nx.from_numpy_array(D)
            length = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    else:
        G = nx.from_numpy_array(W > 0)
        length = dict(nx.all_pairs_shortest_path_length(G))
    inv_dists = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dij = length.get(i, {}).get(j, np.inf)
            inv_dists.append(0.0 if np.isinf(dij) else 1.0 / dij)
    return np.mean(inv_dists) if inv_dists else 0.0


def clustering_coefficient(W_bin):
    G = nx.from_numpy_array((W_bin > 0).astype(int))
    C = nx.average_clustering(G)
    return C


def clustering_coefficient_transitivity(W_bin):
    G = nx.from_numpy_array((W_bin > 0).astype(int))
    return nx.transitivity(G)


def clustering_coefficient_lynn(W, th):
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
    cc = np.sum(triangles) / (np.sum(triplets) + 1e-30)
    return cc


def density(W, th):
    W = np.abs(W) > th
    W = W.astype(float)
    N = W.shape[0]
    return np.sum(W) / (N * (N - 1))


def heterogeneity(W, th=0.0):
    N = W.shape[0]
    I, J = np.triu_indices(N, k=1)
    conn_strengths = W[I, J]
    conn_strengths = conn_strengths[conn_strengths > th]
    # Unique strengths and counts:
    s_values, s_counts = np.unique(conn_strengths, return_counts=True)
    # Normalize the counts to sum to 1:
    p = s_counts / (np.sum(s_counts) + 1e-30)
    # Compute heterogeneity:
    diff_matrix = np.abs(s_values[:, None] - s_values[None, :])  # Matrix of |s_i - s_j|
    P_matrix = np.outer(p, p)  # p_i * p_j
    heterogeneity = 0.5 * np.sum(diff_matrix * P_matrix) / (np.mean(conn_strengths) + 1e-30)
    return heterogeneity


def characteristic_path_length(G):
    if not nx.is_connected(G):
        # Take the largest component
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    try:
        L = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        L = np.inf
    return L


def small_worldness(Glcc, niter=100, nrand=20, max_tries_multiplier=100, seed=42):
    """
        Sigma = (C/Crand) / (L/Lrand) using connected degree-preserving random graphs.
        Omega = (Lrand / L) - (C / Clatt)
        where Lrand, Crand is from connected, degree-preserving randomizations of G;
        Clatt is from ring lattice reference graph
    """
    m = Glcc.number_of_edges()
    n = Glcc.number_of_nodes()
    if m < 4 or n < 3 or not nx.is_connected(Glcc):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    C = nx.transitivity(Glcc)  # global clustering
    L = nx.average_shortest_path_length(Glcc)

    Crand, Lrand = [], []
    # deg = [d for _, d in Glcc.degree()]
    nswap = max(1, int(niter * m))
    # max_tries = max(nswap, int(max_tries_multiplier * m))
    H = Glcc.copy()
    for i in range(nrand):
        nx.connected_double_edge_swap(H, nswap=nswap, seed=rng)
        Crand.append(nx.transitivity(H))
        Lrand.append(characteristic_path_length(H))
    Crand = float(np.mean(Crand)) if len(Crand) else float("nan")
    Lrand = float(np.mean(Lrand)) if len(Lrand) else float("nan")

    sigma = (C / Crand) / (L / Lrand) if Crand > 0 and L > 0 and Lrand > 0 else float("nan")

    # Lattice reference
    avg_deg = 2 * m / n
    k = max(2, int(round(avg_deg)))
    if k % 2 == 1:
        k += 1 if k < avg_deg else -1
    k = max(2, min(k, n - 1))
    if k % 2 == 1:
        k -= 1
    Latt = nx.watts_strogatz_graph(n, k, 0.0)
    Clatt = nx.transitivity(Latt)
    # Llatt = characteristic_path_length(Latt)
    omega = (Lrand / L) - (C / Clatt) if L > 0 and Clatt > 0 and Lrand > 0 else float("nan")

    return sigma, omega


# ---------- Spectral metrics ----------
def laplacian_spectrum(W):
    # Binary or weighted accepted, but Laplacian uses degree matrix from weights
    L = np.diag(W.sum(axis=1)) - W
    evals = np.sort(np.real(linalg.eigvals(L)))
    return evals


def spectral_radius(W):
    # Largest eigenvalue of adjacency (weighted)
    vals = linalg.eigvals(W)
    return float(np.max(np.real(vals)))


def algebraic_connectivity(W):
    evals = laplacian_spectrum(W)
    if len(evals) < 2:
        return 0.0
    return float(evals[1])


# ---------- K-core ----------
def kcore_max_k(W_bin):
    G = nx.from_numpy_array((W_bin > 0).astype(int))
    if G.number_of_edges() == 0:
        return 0
    core_nums = nx.core_number(G)
    return int(max(core_nums.values()))


# ---------- Dynamical thresholds ----------
def sis_threshold(W, mu_over_beta=1.0):
    # SIS threshold condition: (β/μ) * ρ(A) = 1  => threshold = 1/ρ(A)
    rho = spectral_radius(W)
    thr = 1.0 / (rho + 1e-12)
    # If you know β/μ, compute indicator crossing: (β/μ)*rho
    indicator = (1.0 / mu_over_beta) * rho
    return thr, indicator  # thr: needed β/μ; indicator >1 implies endemic regime


def kuramoto_Kc(W, delta=1.0):
    # Kc ≈ 2Δ / λ2(L)
    lam2 = algebraic_connectivity(W)
    if lam2 <= 0:
        return np.inf
    return 2.0 * delta / lam2


def diffusion_mixing_time(W):
    lam2 = algebraic_connectivity(W)
    if lam2 <= 0:
        return np.inf
    return 1.0 / lam2


# ---------- Structure-function coupling proxy ----------
def communicability(W, sigma=1.0):
    # G = exp(sigma * A)
    return linalg.expm(sigma * W)


def structure_function_coupling(W, F, sigma=1.0, method="communicability"):
    # Predict functional from structural proxy and correlate with actual F
    if method == "communicability":
        Cpred = communicability(W, sigma=sigma)
    else:
        # Diffusion kernel: exp(-t L)
        L = np.diag(W.sum(axis=1)) - W
        Cpred = linalg.expm(-sigma * L)
    iu = np.triu_indices_from(W, k=1)
    x = Cpred[iu]
    y = F[iu]
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


# ---------- Avalanche criticality proxies from correlations ----------
def branching_ratio_proxy(R, thr=0.3):
    # Threshold correlation to form adjacency, compute mean degree and normalize
    # This is a crude proxy; real branching needs time-series events.
    A = (np.abs(R) >= thr).astype(float)
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    m = float(np.mean(deg) / (A.shape[0] - 1 + 1e-12))
    return m  # heuristic: closer to 1 suggests denser propagation capacity


# ---------- Change-point detection ----------
def changepoint_analysis(x, y, pen=5.0, model="rbf"):
    # x: list/array of time (sorted); y: order parameter, aligned with x
    # Uses ruptures; returns indices of change-points (end indices)
    # Remove nans
    mask = np.isfinite(y)
    if mask.sum() < 4:
        return None, None

    y = np.asarray(y).reshape(-1, 1)
    algo = rpt.Pelt(model=model).fit(y)
    # penalty can be tuned; larger => fewer change-points
    cp_idx = algo.predict(pen=pen)
    # Convert end indices to ages (exclude last point which is n)
    n = len(x)
    cps = [x[i - 1] for i in cp_idx if i < n]
    return cps, cp_idx


def forman_ricci_curvature(G, edge):
    """
    Calculate Forman-Ricci curvature for a specific edge
    """
    u, v = edge

    # Get degrees of the nodes
    degree_u = G.degree(u)
    degree_v = G.degree(v)

    # Edge weight (default to 1 if unweighted)
    if nx.is_weighted(G):
        edge_weight = G[u][v].get('weight', 1.0)
    else:
        edge_weight = 1.0

        # Calculate Forman-Ricci curvature
    curvature = 4 - degree_u - degree_v

    # Add contribution from adjacent edges (if weighted)
    if nx.is_weighted(G):
        for neighbor in G.neighbors(u):
            if neighbor != v:
                curvature -= G[u][neighbor].get('weight', 1.0) / math.sqrt(
                    edge_weight * G[u][neighbor].get('weight', 1.0))

        for neighbor in G.neighbors(v):
            if neighbor != u:
                curvature -= G[v][neighbor].get('weight', 1.0) / math.sqrt(
                    edge_weight * G[v][neighbor].get('weight', 1.0))

    return curvature


def compute_forman_ricci_curvatures(G):
    """
    Compute Forman-Ricci curvature for all edges in the graph
    """
    curvatures = {}
    for edge in G.edges():
        curvatures[edge] = forman_ricci_curvature(G, edge)
    return curvatures


def forman_ricci_entropy(corr_matrix, threshold=0.5):
    """
    Calculate Forman-Ricci entropy of the graph
    """
    G = nx.Graph()
    n = corr_matrix.shape[0]
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):  # Only upper triangle to avoid duplicates
            if corr_matrix[i, j] >= threshold:  # Elements are connected
                G.add_edge(i, j, weight=corr_matrix[i, j])
                # Get curvatures for all edges
    curvatures = compute_forman_ricci_curvatures(G)
    curvature_values = list(curvatures.values())

    # Shift curvatures to make them all positive (for probability distribution)
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

    return entropy, curvatures


def find_correlated_components(corr_matrix, threshold=0.6, fully_connect=False):
    G = nx.Graph()
    n = corr_matrix.shape[0]
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):  # Only upper triangle to avoid duplicates
            if corr_matrix[i, j] >= threshold:  # Elements are connected
                G.add_edge(i, j, weight=corr_matrix[i, j])

                # Find connected components
    if fully_connect:
        connected_components = nx.find_cliques(G)
    else:
        connected_components = list(nx.connected_components(G))
    return connected_components, G


def get_degree_dist(corr_matrix, threshold=0.6):
    G = nx.Graph()
    n = corr_matrix.shape[0]
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):  # Only upper triangle to avoid duplicates
            if corr_matrix[i, j] >= threshold:  # Elements are connected
                G.add_edge(i, j, weight=corr_matrix[i, j])
    degrees = [d for _, d in G.degree()]
    mean_degree = np.mean(degrees)
    max_degree = max(degrees)
    degree_counts = nx.degree_histogram(G)
    degree_values = range(len(degree_counts))
    degree_prob = [count / sum(degree_counts) for count in degree_counts]
    return degree_values, degree_prob, mean_degree, max_degree, degrees


def get_powerlaw(ss, trange):
    ''' fit exponent to variance curve'''
    logss = np.log(np.abs(ss))
    y = logss[trange][:, np.newaxis]
    trange += 1
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:, np.newaxis], np.ones((nt, 1))), axis=1)
    w = 1.0 / trange.astype(np.float32)[:, np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()

    allrange = np.arange(0, ss.size).astype(int) + 1
    x = np.concatenate((-np.log(allrange)[:, np.newaxis], np.ones((ss.size, 1))), axis=1)
    ypred = np.exp((x * b).sum(axis=1))
    alpha = b[0]
    return alpha, ypred


def power_law(x, a, b, linear=True):
    if linear:
        return a * x + b
    else:
        return b * x ** a


color_array = [
    [0, 77, 128],
    [181, 23, 0],
    [1, 113, 0],
    [242, 112, 0],
    [120, 0, 150],
    [0, 168, 157],
    [203, 41, 123],
    [34, 139, 34],
    [101, 67, 33],
    [135, 206, 235],
    [218, 165, 32],
    [107, 142, 35],
    [138, 154, 91],
    [245, 245, 220],
    [112, 128, 144],
    [183, 200, 153],
    [148, 112, 196],
    [0, 0, 0]
]
color_list = [[r / 255, g / 255, b / 255] for r, g, b in color_array]
plt.rcParams["text.usetex"] = False


def set_figure(num_figure_per_panel, double_panel=True):
    n = num_figure_per_panel
    if double_panel:
        fontsize = 6 * n
    else:
        fontsize = 12 * n
    plt.rcParams['figure.figsize'] = [3.5, 3.5]  # figure size
    plt.rcParams['axes.labelsize'] = fontsize  # For x/y axis labels
    plt.rcParams['axes.titlesize'] = fontsize
    plt.rcParams['xtick.labelsize'] = fontsize  # For x-tick labels
    plt.rcParams['ytick.labelsize'] = fontsize  # For y-tick labels
    plt.rcParams['legend.fontsize'] = fontsize  # For legend
    plt.rcParams['font.size'] = fontsize  # Base font size
    plt.rcParams['font.family'] = 'Arial'
    mpl.rcParams['axes.formatter.useoffset'] = False  # Don't use offset notation (e.g. +1e7)
    mpl.rcParams['axes.formatter.use_mathtext'] = True  # Use '1 × 10^7' style
    mpl.rcParams['axes.formatter.limits'] = (-3, 3)  # Always use scientific notation


def remove_background(ax):
    # Remove panes (backgrounds)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Remove axes lines (make spines invisible)
    ax.xaxis.line.set_color((0, 0, 0, 0))
    ax.yaxis.line.set_color((0, 0, 0, 0))
    ax.zaxis.line.set_color((0, 0, 0, 0))

    # Remove grids
    ax.grid(False)

    # Remove ticks and tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Remove pane borders (the "box" lines)
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')


def np_pearson_corr(x, y):
    """
    Calculate the pearson correlation between two random vectors
    :param x: row = sample, column = observed data
    :param y: same as x
    :return: correlation coefficient matrix with shape (x.shape[0], y.shape[0])
    """
    xv = x - x.mean(axis=1, keepdims=True)
    yv = y - y.mean(axis=1, keepdims=True)
    xvss = (xv * xv).sum(axis=1)
    yvss = (yv * yv).sum(axis=1)
    outer = np.sqrt(np.outer(xvss, yvss))
    outer[outer == 0] = float('inf')
    result = np.matmul(yv, xv.transpose()) / outer
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)


def hist(x, bins=100, bd=None):
    if bd:
        p, e = np.histogram(x, bins=bins, range=bd, density=True)
    else:
        p, e = np.histogram(x, bins=bins, density=True)
    idx = p > 0
    p = p[idx]
    e = 0.5 * (e[:-1] + e[1:])
    e = e[idx]
    p += 1e-15
    p = p / p.sum()
    return p, e


def flatten_params(params):
    return torch.cat([p.view(-1) for p in params])


# Unflatten a vector back into the original parameter shapes
def unflatten_params(flat_params, params):
    """Unflatten flat_params with shape information from params"""
    split_sizes = [torch.numel(p) for p in params]
    split_shapes = [p.shape for p in params]
    split_params = flat_params.split(split_sizes)
    return [p.view(shape) for p, shape in zip(split_params, split_shapes)]


def stable_gibbs_measure(beta, energies):
    energies[np.isnan(energies) | np.isinf(energies)] = 1e10  # handling gradient explosion
    log_weights = -beta * energies

    # Shift by the maximum log weight
    max_log_weight = np.max(log_weights)
    shifted_log_weights = log_weights - max_log_weight

    # For states with extremely low probabilities, set to effectively zero
    significant_states = shifted_log_weights > -30  # log(1e-30)

    # Only exponentiate the significant states
    weights = np.zeros_like(energies, dtype=float) + 1e-30
    weights[significant_states] = np.exp(shifted_log_weights[significant_states])

    # Normalize
    Z = np.sum(weights)
    probabilities = weights / Z
    probabilities[probabilities < 1e-30] = 1e-30
    probabilities /= np.sum(probabilities)

    return probabilities


# For log partition function
def log_partition_function(beta, energies):
    c = -beta * np.min(energies)  # The constant term
    return c + np.log(np.sum(np.exp(-beta * energies - c)))


def get_temperature(lk, b, temp_range=(-1e10, 1e10), tol=1e-6, prior_guess=0):
    t1, t2, = temp_range

    def f(beta):
        p = stable_gibbs_measure(beta, lk)
        return np.sum(p * lk) - b

    if f(t1) * f(t2) > 0:
        if b > np.mean(lk):
            temp = t1
        else:
            temp = t2
    else:
        result = optimize.root_scalar(f, bracket=[t1, t2], x0=prior_guess, xtol=tol, method='brentq')
        temp = result.root
    return temp


def quasi_entropy(lk, temp):
    p = stable_gibbs_measure(temp, lk)
    return -np.sum(p * np.log(p))


def norm_dist_p(p, e):
    idx = p > 0
    p = p[idx]
    e = e[1:]
    e = e[idx]
    p += 1e-10
    p = p / p.sum()
    return p, e


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return torch.tensor([torch.sum(x * y) for (x, y) in zip(xs, ys)]).sum()


def group_add(params, update, alpha):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :param alpha:
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s ** 0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    return hv


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)


def vectorize_weights(model):
    with torch.no_grad():
        if isinstance(model, tuple) or isinstance(model, list):
            g = [p.detach().cpu().numpy().reshape(-1) for p in model if p.requires_grad]
        elif isinstance(model, torch.nn.Module):
            g = [p.data.detach().cpu().numpy().reshape(-1) for p in model.parameters() if p.requires_grad]
        return np.concatenate(g, axis=0)


def vectorize_grads(model):
    with torch.no_grad():
        if isinstance(model, tuple) or isinstance(model, list):
            g = [p.detach().cpu().numpy().reshape(-1) for p in model if p.requires_grad]
        elif isinstance(model, torch.nn.Module):
            g = [p.grad.data.detach().cpu().numpy().reshape(-1) for p in model.parameters() if p.requires_grad]
    return np.concatenate(g, axis=0)


def restore_weights(vec_weights, neurons):
    neurons = int(neurons)
    params = []
    d = 784 * neurons
    params.append(vec_weights[:d].reshape(neurons, 784))
    d1 = d + neurons
    params.append(vec_weights[d:d1])
    d = d1
    d1 = d1 + neurons * neurons
    params.append(vec_weights[d:d1].reshape(neurons, neurons))
    d = d1
    d1 = d1 + neurons
    params.append(vec_weights[d:d1])
    d = d1
    d1 = d1 + 10 * neurons
    params.append(vec_weights[d:d1].reshape(10, neurons))
    params.append(vec_weights[d1:])
    return params


def restore_weights_like(vec_weights, ref_stat_dict):
    stat_dict = {}
    sizes = [v.numel() for k, v in ref_stat_dict.items()]
    vs = np.split(vec_weights, np.cumsum(sizes))[:-1]
    for i, (k, v) in enumerate(ref_stat_dict.items()):
        stat_dict[k] = torch.from_numpy(vs[i].reshape(v.shape))
    return stat_dict


# def point_reducer(x, reducer):
#     if x.ndim == 1:
#         x = x.reshape(1, -1)
#     return reducer.transform(x)


# def vector_reducer(vec, reducer):
#     # original point is also transformed
#     if vec.ndim == 1:
#         x0 = np.zeros(vec.shape[0]).reshape(1, -1)
#         x1 = vec.reshape(1, -1)
#     else:
#         x0 = np.zeros(vec.shape[1]).reshape(1, -1)
#         x1 = vec
#     x0 = reducer.transform(x0)
#     x1 = reducer.transform(x1)
#     return x1 - x0


def reduce_weights(model, reducer):
    x = vectorize_weights(model)
    return (x - reducer.mean_) @ reducer.components_.T


def reduce_grads(model, reducer):
    x = vectorize_grads(model)
    return x @ reducer.components_.T


def scale_to_range(arr, range=(0, 1)):
    a, b = range
    arr = np.asarray(arr)  # Ensure input is a NumPy array
    min_val = arr.min()
    max_val = arr.max()
    # Avoid division by zero
    if min_val == max_val:
        return np.full_like(arr, a)
    scaled = (b - a) * (arr - min_val) / (max_val - min_val) + a
    zeros = (b - a) * (0 - min_val) / (max_val - min_val) + a
    return scaled, zeros
