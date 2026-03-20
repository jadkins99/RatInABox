import numpy as np

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

