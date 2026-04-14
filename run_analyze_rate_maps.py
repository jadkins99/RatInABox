from utils import load_rate_maps
from representation_analysis import build_peak_dataset
from plotting import plot_peaks_per_environment, plot_rsm_decay
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np


# =========================================================
# 1. Convert rate maps → population vectors per spatial bin
# =========================================================
def get_population_vectors(rate_maps):
    """
    rate_maps: (n_units, H, W)

    returns:
    pop_vecs: (H, W, n_units)
    """
    n_units, H, W = rate_maps.shape

    pop_vecs = np.zeros((H, W, n_units))

    for u in range(n_units):
        pop_vecs[:, :, u] = rate_maps[u]

    return pop_vecs


# =========================================================
# 2. Normalize (VERY IMPORTANT for fair comparison)
# =========================================================
def normalize_pop_vectors(pop_vecs):
    H, W, n_units = pop_vecs.shape

    flat = pop_vecs.reshape(-1, n_units)

    flat = (flat - flat.mean(axis=1, keepdims=True)) / (
        flat.std(axis=1, keepdims=True) + 1e-8
    )

    return flat.reshape(H, W, n_units)


# =========================================================
# 3. Compute pixel-wise RSM
# =========================================================
def spatial_rsm(pop_vecs):
    """
    pop_vecs: (H, W, n_units)

    returns:
    rsm: (H*W, H*W)
    """
    H, W, n_units = pop_vecs.shape

    flat = pop_vecs.reshape(H * W, n_units)

    n_bins = H * W
    rsm = np.zeros((n_bins, n_bins))

    for i in range(n_bins):
        for j in range(n_bins):

            rsm[i, j] = np.corrcoef(flat[i], flat[j])[0, 1]

    return rsm


# =========================================================
# 4. Spatial distance matrix
# =========================================================
def spatial_distance_matrix(H, W):
    coords = [(i, j) for i in range(H) for j in range(W)]
    n = len(coords)

    dist = np.zeros((n, n))

    for i, (x1, y1) in enumerate(coords):
        for j, (x2, y2) in enumerate(coords):
            dist[i, j] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    return dist


# =========================================================
# 5. Extract decay curve
# =========================================================
def rsm_decay_curve(rsm, dist):
    upper = np.triu_indices_from(rsm, k=1)

    d_vals = dist[upper]
    r_vals = rsm[upper]

    return d_vals, r_vals


# =========================================================
# 6. Bin + average decay
# =========================================================
def bin_decay(d, r, n_bins=20):
    bins = np.linspace(0, np.max(d), n_bins + 1)

    digitized = np.digitize(d, bins)

    means = []
    centers = []

    for i in range(1, len(bins)):
        mask = digitized == i
        if np.sum(mask) > 0:
            means.append(np.mean(r[mask]))
            centers.append(bins[i])

    return np.array(centers), np.array(means)


# =========================================================
# 7. Full pipeline per model/env
# =========================================================
def compute_rsm_decay(rate_maps):
    """
    rate_maps: (n_units, H, W)
    """

    H, W = rate_maps.shape[1:]

    pop_vecs = get_population_vectors(rate_maps)
    pop_vecs = normalize_pop_vectors(pop_vecs)

    rsm = spatial_rsm(pop_vecs)
    dist = spatial_distance_matrix(H, W)

    d, r = rsm_decay_curve(rsm, dist)

    return bin_decay(d, r)


print("Loading rate maps...")
data = load_rate_maps("data")

peaks = build_peak_dataset(data)
print("Plotting peaks per environment...")
plot_peaks_per_environment(peaks)

print("Analyzing rate map spatial structure...")

env = "env_walls"

model_results = {}

for model in data:
    print(model)

    all_maps = data[model][env]  # list of runs

    # average across runs first
    mean_map = np.mean(np.stack(all_maps), axis=0)

    x, y = compute_rsm_decay(mean_map)

    model_results[model] = (x, y)

plot_rsm_decay(model_results)