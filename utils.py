# utils.py
import os
import pickle
import numpy as np

def save_data(data, filepath):
    """
    Save data to a pickle file.

    Args:
        data:     any Python object to save
        filepath: full path including filename and .pkl extension
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_data(filepath):
    """
    Load data from a pickle file.

    Args:
        filepath: full path including filename and .pkl extension

    Returns:
        the loaded object
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)

def load_runs_bins(data_dir, env_shape):
    """
    Load runs_bins from saved numpy files.

    Args:
        data_dir:   base data directory (e.g., "data")
        env_shape:  environment name (used in folder name)

    Returns:
        runs_bins: list of runs (each run is episodes × timesteps × features)
    """

    env_path = os.path.join(data_dir, f"env_{env_shape}")
    runs_bins = []

    # sort to keep seeds in order
    seed_folders = sorted([
        f for f in os.listdir(env_path)
        if f.startswith("seed_")
    ])

    for seed_folder in seed_folders:
        seed_path = os.path.join(env_path, seed_folder)

        # extract seed number
        seed_num = seed_folder.split("_")[-1]

        file_path = os.path.join(
            seed_path,
            f"all_episodes_bins_seed_{seed_num}"
        )

        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        data = np.load(file_path, allow_pickle=True)
        runs_bins.append(data)

    return runs_bins