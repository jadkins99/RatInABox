import numpy as np
from scipy.ndimage import gaussian_filter1d

def bootstrap_ci(arr, n_boot=2000, alpha=0.05, seed=42):
    arr = np.asarray(arr)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True)
    stats = np.mean(boots, axis=1)

    low = np.percentile(stats, 100 * alpha / 2)
    high = np.percentile(stats, 100 * (1 - alpha / 2))
    mean = np.mean(arr)

    return mean, low, high

def compute_sparsity_per_timestep_single(out_arrays, thres=0.1):
    """
    Computes the percentage of inactive (dead) features per timestep for a single run.

    Args:
        out_arrays: list[episodes][timesteps] of FTA output vectors
        thres:      threshold below which a feature is considered inactive

    Returns:
        dead_percent: array of % inactive features per timestep
    """
    all_timesteps = []
    for ep_out in out_arrays:
        all_timesteps.extend(ep_out)

    dead_percent = []
    for timestep_values in all_timesteps:
        timestep_values = np.array(timestep_values)
        inactive = timestep_values < thres
        dead_percent.append(inactive.mean() * 100)

    return np.array(dead_percent)

def compute_sparsity_per_episode_single(out_arrays, thres=0.1):
    """
    Computes the percentage of inactive features per episode for a single run.

    Args:
        out_arrays: list[episodes][timesteps] of FTA output vectors
        thres:      threshold below which a feature is considered inactive

    Returns:
        sparsity_per_episode: array of % inactive features per episode
    """
    sparsity_per_episode = []

    for ep_out in out_arrays:
        ep_out = np.array(ep_out)  # (timesteps, features)
        inactive = ep_out < thres
        percent_timestep = inactive.mean(axis=1) * 100  # % per timestep
        percent_episode = percent_timestep.mean()        # average across timesteps
        sparsity_per_episode.append(percent_episode)

    return np.array(sparsity_per_episode)


def compute_sparsity_per_timestep(out_arrays, thres=0.1):
    """
    Computes the mean and standard error of sparsity per timestep across runs.

    Returns:
        mean_dead: np.array (timesteps,)
        se_dead:   np.array (timesteps,)
    """
    # Flatten episodes per run
    flattened_runs = []
    for run in out_arrays:
        run_timesteps = []
        for ep_out in run:
            run_timesteps.extend(ep_out)
        flattened_runs.append(run_timesteps)

    max_timesteps = max(len(r) for r in flattened_runs)

    mean_dead = []
    se_dead = []

    for t in range(max_timesteps):
        per_run_values = []

        for run in flattened_runs:
            if t < len(run):
                timestep_values = np.array(run[t])
                inactive = timestep_values < thres
                percent = inactive.mean() * 100
                per_run_values.append(percent)

        per_run_values = np.array(per_run_values)

        mean_dead.append(np.mean(per_run_values))

        if len(per_run_values) > 1:
            se = np.std(per_run_values, ddof=1) / np.sqrt(len(per_run_values))
        else:
            se = 0.0

        se_dead.append(se)

    return np.array(mean_dead), np.array(se_dead)


def compute_sparsity_per_episode(runs_out_arrays, thres=0.1):
    """
    Returns:
        mean, ci_low, ci_high per episode
    """

    max_episodes = max(len(run) for run in runs_out_arrays)

    # collect per episode across runs
    per_episode = [[] for _ in range(max_episodes)]

    for run in runs_out_arrays:
        for ep in range(len(run)):
            ep_out = np.array(run[ep])  # (timesteps, features)

            inactive = ep_out < thres
            percent_timestep = inactive.mean(axis=1) * 100
            percent_episode = percent_timestep.mean()

            per_episode[ep].append(percent_episode)

    means, lows, highs = [], [], []

    for ep_values in per_episode:
        ep_values = np.array(ep_values)

        m, lo, hi = bootstrap_ci(ep_values)

        means.append(m)
        lows.append(lo)
        highs.append(hi)

    return np.array(means), np.array(lows), np.array(highs)

def compute_bin_counts_per_timestep(runs_out_arrays, num_bins, threshold=0.1):
    """
    Compute bin activation counts (> threshold) for each timestep,
    averaged across runs.

    Args:
        runs_out_arrays: list[runs][episodes][timesteps] of flattened outputs
                         Each timestep: shape (n_units,)
        num_bins:        number of bins per unit (e.g. 11 for FTA)
        threshold:       threshold above which a feature is active

    Returns:
        bin_counts: np.array of shape (timesteps, num_bins)
    """

    # Flatten episodes per run → list[runs][timesteps]
    flattened_runs = []
    for run in runs_out_arrays:
        run_timesteps = []
        for ep in run:
            run_timesteps.extend(ep)
        flattened_runs.append(run_timesteps)

    max_timesteps = max(len(r) for r in flattened_runs)

    all_counts = []

    for t in range(max_timesteps):
        timestep_counts = []

        for run in flattened_runs:
            if t < len(run):
                arr = np.array(run[t])  # shape: (n_units,)

                # reconstruct bins
                if arr.ndim != 1 or arr.size % num_bins != 0:
                    raise ValueError(
                        f"Cannot reshape array of shape {arr.shape} into (-1, {num_bins})"
                    )

                arr = arr.reshape(-1, num_bins)  # (num_outputs, num_bins)

                # Threshold → active bins
                active = arr > threshold

                # Count active per bin
                bin_counts = np.sum(active, axis=0)

                timestep_counts.append(bin_counts)

        # Average across runs
        all_counts.append(np.mean(timestep_counts, axis=0))

    return np.array(all_counts)  # (timesteps, num_bins)


def compute_bin_counts_per_timestep_single(out_arrays, num_bins, threshold=0.1):
    """
    Compute bin activation counts (> threshold) for each timestep for a single run.

    Args:
        out_arrays: list[episodes][timesteps] of flattened FTA outputs
                    Each timestep: shape (num_outputs * num_bins,)
        num_bins:   number of bins per feature (IMPORTANT: use actual FTA num_tiles)
        threshold:  threshold above which a feature is considered active

    Returns:
        bin_counts: np.array of shape (timesteps, num_bins)
    """

    # Flatten episodes into a single sequence of timesteps
    all_timesteps = []
    for ep in out_arrays:
        all_timesteps.extend(ep)

    all_counts = []

    for timestep_values in all_timesteps:
        arr = np.array(timestep_values)

        # ---- reconstruct bins ----
        if arr.ndim == 1:
            if arr.size % num_bins != 0:
                raise ValueError(
                    f"Cannot reshape array of size {arr.size} into (-1, {num_bins})"
                )
            arr = arr.reshape(-1, num_bins)  # (num_outputs, num_bins)

        elif arr.ndim == 2:
            # already structured → do nothing
            pass

        else:
            raise ValueError(f"Unexpected array shape: {arr.shape}")

        # ---- compute active bins ----
        active = arr > threshold
        bin_counts = np.sum(active, axis=0)  # sum across outputs

        all_counts.append(bin_counts)

    return np.array(all_counts)  # shape: (timesteps, num_bins)


def compute_rate_maps_single(
    states,
    out_arrays,
    n_spatial_bins=15,
    filter_size=1.5,
    obstacles=None
):
    """
    Rate maps using consistent (y, x) indexing.
    Assumes environment is [0,1] x [0,1].
    """

    # =========================
    # Flatten
    # =========================
    position = np.array([s for ep in states for s in ep])
    trace    = np.array([o for ep in out_arrays for o in ep])

    n_units = trace.shape[1]
    T = trace.shape[0]

    rate_maps = np.zeros((n_units, n_spatial_bins, n_spatial_bins))
    count_map = np.zeros((n_spatial_bins, n_spatial_bins))

    # =========================
    # Bin positions (NO swapping issues)
    # =========================
    position = np.clip(position, 0.0, 1.0)

    position_binned = (position * n_spatial_bins).astype(int)
    position_binned = np.clip(position_binned, 0, n_spatial_bins - 1)

    # =========================
    # ACCUMULATION (FIXED: y, x order)
    # =========================
    for t, (x, y) in enumerate(position_binned):
        rate_maps[:, y, x] += trace[t]
        count_map[y, x] += 1

    # =========================
    # Occupancy mask
    # =========================
    nan_map = np.full_like(count_map, np.nan)
    nan_map[count_map > 0] = 1.0

    # =========================
    # Obstacles (FIXED indexing)
    # =========================
    if obstacles is not None and len(obstacles) > 0:
        bin_edges = np.linspace(0, 1, n_spatial_bins + 1)

        for obs in obstacles:
            for ix in range(n_spatial_bins):
                for iy in range(n_spatial_bins):

                    bin_x_min, bin_x_max = bin_edges[ix], bin_edges[ix + 1]
                    bin_y_min, bin_y_max = bin_edges[iy], bin_edges[iy + 1]

                    if (bin_x_min <= obs['x_max'] and bin_x_max >= obs['x_min'] and
                        bin_y_min <= obs['y_max'] and bin_y_max >= obs['y_min']):
                        nan_map[iy, ix] = np.nan

    # =========================
    # Normalize
    # =========================
    rate_maps = rate_maps / np.where(count_map > 0, count_map, 1)

    # =========================
    # Smooth
    # =========================
    if filter_size:
        rate_maps = gaussian_filter1d(rate_maps, sigma=filter_size, axis=1)
        rate_maps = gaussian_filter1d(rate_maps, sigma=filter_size, axis=2)

    # =========================
    # Apply mask
    # =========================
    for u in range(n_units):
        rate_maps[u] *= nan_map

    occupancy_map = count_map / T

    return rate_maps, occupancy_map


def compute_dead_neurons_with_se(runs_out_arrays, thres=0.1):
    """
    Computes the mean and standard error of neurons that are
    inactive for the entire episode.

    A neuron is considered "dead" in an episode if it is below
    threshold at ALL timesteps.

    Args:
        runs_out_arrays: list[runs][episodes][timesteps][features]
        thres: threshold below which a feature is considered inactive

    Returns:
        mean_dead: np.array (episodes,)
        se_dead:   np.array (episodes,)
    """

    max_episodes = max(len(run) for run in runs_out_arrays)

    dead_percent_runs = []

    for run in runs_out_arrays:
        run_episode_values = []

        for ep in range(len(run)):
            ep_out = np.array(run[ep])  # (timesteps, features)

            # True if neuron is inactive at ALL timesteps
            dead_neurons = (ep_out < thres).all(axis=0)

            # percentage of dead neurons
            percent_dead = dead_neurons.mean() * 100

            run_episode_values.append(percent_dead)

        dead_percent_runs.append(run_episode_values)

    mean_dead = []
    se_dead = []

    for ep in range(max_episodes):
        ep_values = [run[ep] for run in dead_percent_runs if ep < len(run)]
        ep_values = np.array(ep_values)

        mean_dead.append(np.mean(ep_values))

        if len(ep_values) > 1:
            se = np.std(ep_values, ddof=1) / np.sqrt(len(ep_values))
        else:
            se = 0.0

        se_dead.append(se)

    return np.array(mean_dead), np.array(se_dead)

def compute_dead_neurons(runs_out_arrays, thres=0.1):
    """
    Returns:
        mean, ci_low, ci_high per episode
    """

    max_episodes = max(len(run) for run in runs_out_arrays)

    per_episode = [[] for _ in range(max_episodes)]

    for run in runs_out_arrays:
        for ep in range(len(run)):
            ep_out = np.array(run[ep])  # (timesteps, features)

            dead_neurons = (ep_out < thres).all(axis=0)
            percent_dead = dead_neurons.mean() * 100

            per_episode[ep].append(percent_dead)

    means, lows, highs = [], [], []

    for ep_values in per_episode:
        ep_values = np.array(ep_values)

        m, lo, hi = bootstrap_ci(ep_values)

        means.append(m)
        lows.append(lo)
        highs.append(hi)

    return np.array(means), np.array(lows), np.array(highs)

def get_active_bins_from_run_bins(run_bins, threshold=0.99):
    """
    For each timestep, find which bin has a value of 1 (fully active) 
    using the already-computed per-dimension bin arrays.

    Args:
        run_bins:  list[runs][episodes][timesteps][input_dim] of np.array of shape (n_tiles,)
        threshold: threshold to consider a bin as fully active (default 0.99)

    Returns:
        all_active_bins: list[runs][episodes][timesteps][input_dim] of active bin indices
    """
    all_active_bins = []

    for run in run_bins:
        run_active = []
        for episode in run:
            ep_active = []
            for timestep_bins in episode:
                timestep_active = []
                for dim_bins in timestep_bins:
                    active = np.where(np.array(dim_bins) >= threshold)[0]
                    timestep_active.append(active.tolist() if len(active) > 0 else None)
                ep_active.append(timestep_active)
            run_active.append(ep_active)
        all_active_bins.append(run_active)

    return all_active_bins

def check_active_bins_below_threshold(active_bins, bin_threshold=10):
    """
    Check if any active bin (value ~1) is below a given bin index threshold.

    Args:
        active_bins:    list[runs][episodes][timesteps][input_dim] of active bin indices
        bin_threshold:  bin index below which we flag as unexpected (default 10)

    Returns:
        violations: list of tuples (run, episode, timestep, dim, bin_idx) where a bin
                    below bin_threshold is fully active
    """
    violations = []

    for run_idx, run in enumerate(active_bins):
        for ep_idx, episode in enumerate(run):
            for t_idx, timestep in enumerate(episode):
                for dim_idx, bins in enumerate(timestep):
                    if bins is None:
                        continue
                    for bin_idx in bins:
                        if bin_idx < bin_threshold:
                            violations.append((run_idx, ep_idx, t_idx, dim_idx, bin_idx))

    return violations