import numpy as np
from scipy.ndimage import gaussian_filter1d

def compute_dead_neurons_per_timestep(runs_out_arrays, thres=0.1):
    # Flatten episodes per run
    flattened_runs = []
    for run in runs_out_arrays:
        run_timesteps = []
        for ep_out in run:
            run_timesteps.extend(ep_out)
        flattened_runs.append(run_timesteps)
    
    # Find maximum number of timesteps across runs
    max_timesteps = max(len(r) for r in flattened_runs)
    
    # Compute percentage of dead neurons per timestep, averaging only over available runs
    dead_percent = []
    for t in range(max_timesteps):
        timestep_values = []
        for run in flattened_runs:
            if t < len(run):
                timestep_values.append(run[t])
        
        timestep_values = np.array(timestep_values)  # shape: (available_runs, n_features)
        inactive = timestep_values < thres
        percent_inactive = inactive.mean(axis=1) * 100  # % per run
        dead_percent.append(percent_inactive.mean())    # average across available runs
    
    return np.array(dead_percent)


def compute_dead_neurons_per_episode(runs_out_arrays, thres=0.1):
    """
    Computes the percentage of inactive neurons per episode.
    
    Procedure:
        1. Compute inactivity per timestep
        2. Average across timesteps within each episode
        3. Average across runs (only over runs that have data at that episode)

    Args:
        runs_out_arrays: list[runs][episodes][timesteps][features]
        thres: threshold below which a feature is considered inactive

    Returns:
        mean_dead_percent: array (% dead neurons per episode averaged across available runs)
    """

    max_episodes = max(len(run) for run in runs_out_arrays)

    dead_percent_runs = []

    for run in runs_out_arrays:
        run_episode_values = []

        for ep in range(len(run)):
            ep_out = np.array(run[ep])  # shape: (timesteps, features)
            inactive = ep_out < thres
            percent_timestep = inactive.mean(axis=1) * 100
            percent_episode = percent_timestep.mean()
            run_episode_values.append(percent_episode)

        dead_percent_runs.append(run_episode_values)

    # average per episode only over runs that have data at that episode
    mean_dead_percent = []
    for ep in range(max_episodes):
        ep_values = [run[ep] for run in dead_percent_runs if ep < len(run)]
        mean_dead_percent.append(np.mean(ep_values))

    return np.array(mean_dead_percent)

def compute_bin_counts_per_timestep(runs_out_arrays, num_bins, threshold=0.1):
    """
    Compute bin counts (> threshold) for each timestep, averaged across available runs.
    
    Returns:
        bin_counts : np.array of shape (timesteps, num_bins)
    """

    # Flatten episodes per run
    flattened_runs = []
    for run in runs_out_arrays:
        run_timesteps = []
        for ep in run:
            run_timesteps.extend(ep)
        flattened_runs.append(run_timesteps)

    # Find maximum number of timesteps across runs
    max_timesteps = max(len(r) for r in flattened_runs)

    all_counts = []

    for t in range(max_timesteps):
        timestep_counts = []

        for run in flattened_runs:
            if t < len(run):
                vec = np.array(run[t])
                bins = np.array_split(vec, num_bins)
                counts = [np.sum(b > threshold) for b in bins]
                timestep_counts.append(counts)

        # average over available runs
        all_counts.append(np.mean(timestep_counts, axis=0))

    return np.array(all_counts)  # shape: (timesteps, num_bins)


def _compute_single_rate_map(position, trace, n_bins, filter_size, buffer):
    """
    Helper to compute rate map for a single set of positions and traces.

    Args:
        position:    np.array of shape (timesteps, 2)
        trace:       np.array of shape (timesteps, n_units)
        n_bins:      number of spatial bins
        filter_size: sigma for gaussian smoothing (None = no smoothing)
        buffer:      buffer for binning

    Returns:
        rate_maps:     np.array of shape (n_units, n_bins, n_bins)
        occupancy_map: np.array of shape (n_bins, n_bins)
        average_events: np.array of shape (n_units,)
    """
    len_recording = trace.shape[0]
    n_units = trace.shape[1]

    rate_maps = np.zeros([n_units, n_bins, n_bins])
    count_map = np.zeros([n_bins, n_bins])

    position_binned = (position // ((np.nanmax(position, axis=0) + buffer) / n_bins)).astype(int)
    position_binned = np.clip(position_binned, 0, n_bins - 1)

    for t, (x, y) in enumerate(position_binned):
        rate_maps[:, x, y] += trace[t, :]
        count_map[x, y] += 1

    nan_map = np.zeros_like(count_map) * np.nan
    nan_map[np.where(count_map)[0], np.where(count_map)[1]] = 1.0

    if filter_size is not None:
        rate_maps = gaussian_filter1d(gaussian_filter1d(rate_maps, sigma=filter_size, axis=1),
                                      sigma=filter_size, axis=2)
        count_map = gaussian_filter1d(gaussian_filter1d(count_map, sigma=filter_size, axis=0),
                                      sigma=filter_size, axis=1)

    for unit in range(n_units):
        rate_maps[unit] = (rate_maps[unit] / count_map) * nan_map

    occupancy_map  = count_map / len_recording
    average_events = np.nanmean(trace, axis=0)

    return rate_maps, occupancy_map, average_events


def compute_fta_rate_maps(runs_states, runs_out_arrays, n_bins=15, filter_size=None, buffer=1e-5):
    """
    Create rate maps from agent position and FTA output data.

    Args:
        runs_states:     list[runs][episodes][timesteps] of agent (x,y) positions
        runs_out_arrays: list[runs][episodes][timesteps] of FTA output vectors
        n_bins:          number of spatial bins in x- and y-dimensions
        filter_size:     sigma size (bin number) for gaussian smoothing (None = no smoothing)
        buffer:          buffer size for rounding binned position data

    Returns:
        rate_maps_per_run:  list of np.array of shape (n_units, n_bins, n_bins) — one per run
        occupancy_per_run:  list of np.array of shape (n_bins, n_bins) — one per run
        rate_maps_avg:      np.array of shape (n_units, n_bins, n_bins) — average across runs
        occupancy_map:      np.array of shape (n_bins, n_bins) — average across runs
        average_events:     np.array of shape (n_units,) — average activation per unit
    """

    # --- Per-run rate maps ---
    rate_maps_per_run  = []
    occupancy_per_run  = []

    for run_s, run_o in zip(runs_states, runs_out_arrays):
        all_positions = []
        all_outputs   = []
        for ep_s, ep_o in zip(run_s, run_o):
            all_positions.extend(ep_s)
            all_outputs.extend(ep_o)

        position = np.array(all_positions)
        trace    = np.array(all_outputs)
        rate_maps, occupancy, _ = _compute_single_rate_map(position, trace, n_bins, filter_size, buffer)
        rate_maps_per_run.append(rate_maps)
        occupancy_per_run.append(occupancy)

    # --- Average across all runs ---
    all_positions = []
    all_outputs   = []
    for run_s, run_o in zip(runs_states, runs_out_arrays):
        for ep_s, ep_o in zip(run_s, run_o):
            all_positions.extend(ep_s)
            all_outputs.extend(ep_o)

    position = np.array(all_positions)
    trace    = np.array(all_outputs)
    rate_maps_avg, occupancy_map, average_events = _compute_single_rate_map(
        position, trace, n_bins, filter_size, buffer
    )

    return rate_maps_per_run, occupancy_per_run, rate_maps_avg, occupancy_map, average_events