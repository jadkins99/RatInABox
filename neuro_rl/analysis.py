import numpy as np

def compute_dead_neurons_per_timestep(runs_out_arrays, thres=0.1):
    """
    Computes the average percentage of inactive (dead) features per timestep
    across multiple runs, truncating to the minimum number of timesteps.

    Args:
        runs_out_arrays: list of runs, each run is a list of episodes,
                         each episode is a list of out_arrays per timestep
        thres: threshold below which a feature is considered inactive

    Returns:
        mean_dead_percent: array of mean % inactive features per timestep
    """
    # Flatten episodes per run
    flattened_runs = []
    for run in runs_out_arrays:
        # concatenate all timesteps from all episodes in the run
        run_timesteps = []
        for ep_out in run:
            run_timesteps.extend(ep_out)  # ep_out is list of arrays per timestep
        flattened_runs.append(run_timesteps)
    
    # Find minimum number of timesteps across runs
    min_timesteps = min(len(r) for r in flattened_runs)
    
    # Truncate and compute percentage of dead neurons per timestep
    dead_percent = []
    for t in range(min_timesteps):
        timestep_values = np.array([run[t] for run in flattened_runs])  # shape: (num_runs, n_features)
        inactive = timestep_values < thres
        percent_inactive = inactive.mean(axis=1) * 100  # % per run
        dead_percent.append(percent_inactive.mean())   # average across runs
    
    return np.array(dead_percent)


def compute_dead_neurons_per_episode(runs_out_arrays, thres=0.1):
    """
    Computes the percentage of inactive neurons per episode.
    
    Procedure:
        1. Compute inactivity per timestep
        2. Average across timesteps within each episode
        3. Average across runs

    Args:
        runs_out_arrays: list[runs][episodes][timesteps][features]
        thres: threshold below which a feature is considered inactive

    Returns:
        mean_dead_percent: array (% dead neurons per episode averaged across runs)
    """

    num_runs = len(runs_out_arrays)

    # find minimum number of episodes across runs
    min_episodes = min(len(run) for run in runs_out_arrays)

    dead_percent_runs = []

    for run in runs_out_arrays:
        run_episode_values = []

        for ep in range(min_episodes):
            ep_out = run[ep]  # list of timestep feature vectors

            ep_out = np.array(ep_out)  # shape: (timesteps, features)

            inactive = ep_out < thres

            # percentage inactive per timestep
            percent_timestep = inactive.mean(axis=1) * 100

            # average across timesteps
            percent_episode = percent_timestep.mean()

            run_episode_values.append(percent_episode)

        dead_percent_runs.append(run_episode_values)

    dead_percent_runs = np.array(dead_percent_runs)

    # average across runs
    mean_dead_percent = dead_percent_runs.mean(axis=0)

    return mean_dead_percent





