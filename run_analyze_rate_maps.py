from utils import load_rate_maps
from representation_analysis import build_peak_dataset
from plotting import plot_peaks_per_environment, plot_pv_matrix_pairwise
import numpy as np


def rate_maps_to_population(rate_maps):
    """
    rate_maps: (n_units, H, W)

    returns:
    (H*W, n_units)
    """
    n_units, H, W = rate_maps.shape
    return rate_maps.reshape(n_units, -1).T

def normalize_population(pop):
    return (pop - pop.mean(axis=1, keepdims=True)) / (
        pop.std(axis=1, keepdims=True) + 1e-8
    )

def compute_pv_matrix(rate_maps_A, rate_maps_B):
    """
    rate_maps_A: (n_units, H, W)
    rate_maps_B: (n_units, H, W)

    Returns:
        (H*W, H*W) pixel-wise correlation matrix
    """

    n_units, H, W = rate_maps_A.shape
    n_bins = H * W

    # Flatten
    A = rate_maps_A.reshape(n_units, n_bins).T  # (n_bins, n_units)
    B = rate_maps_B.reshape(n_units, n_bins).T

    pv_matrix = np.zeros((n_bins, n_bins))

    for i in range(n_bins):
        for j in range(n_bins):

            a = A[i]
            b = B[j]

            # valid neurons only (ignore NaNs)
            mask = (~np.isnan(a)) & (~np.isnan(b))

            if np.sum(mask) > 1:
                pv_matrix[i, j] = np.corrcoef(a[mask], b[mask])[0, 1]
            else:
                pv_matrix[i, j] = np.nan

    return pv_matrix

def compute_pv_all_envs(data_model):
    envs = sorted(list(data_model.keys()))

    # average across runs
    mean_maps = {
        env: np.mean(np.stack(data_model[env]), axis=0)
        for env in envs
    }

    example = next(iter(mean_maps.values()))
    H, W = example.shape[1:]
    print(f"Rate maps shape: {example.shape} (n_units, H, W)")

    for env in mean_maps:
        print(env, mean_maps[env].shape)

    pv_matrix = np.zeros((len(envs), len(envs), H*W, H*W))

    for i, env1 in enumerate(envs):
        for j, env2 in enumerate(envs):

            pv_matrix[i, j] = compute_pv_matrix(
                mean_maps[env1],
                mean_maps[env2]
            )

    return envs, pv_matrix

print("Loading rate maps...")
data = load_rate_maps("data")

peaks = build_peak_dataset(data)
print("Plotting peaks per environment...")
plot_peaks_per_environment(peaks)


print("Computing and plotting PV matrices...")

for model in data:

    print(f"Processing {model}")

    envs, pv_matrix = compute_pv_all_envs(data[model])

    plot_pv_matrix_pairwise(
        envs,
        pv_matrix,
        model_name=model,
        save_dir="figures/pv_matrices", 
        vmin=-0.1,
        vmax=1.0
    )