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

def load_runs_out_arrays(data_dir, env_shape):
    """
    Load runs_out_arrays from saved numpy files.

    Args:
        data_dir:   base data directory (e.g., "data")
        env_shape:  environment name (used in folder name)

    Returns:
        runs_out_arrays: list of runs (each run is episodes x timesteps x features)
    """

    env_path = os.path.join(data_dir, f"env_{env_shape}")
    runs_out_arrays = []

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
            f"all_out_arrays_seed_{seed_num}"
        )

        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        data = np.load(file_path, allow_pickle=True)
        runs_out_arrays.append(data)

    return runs_out_arrays


def load_rate_maps(root="data"):
    data = {}

    for model in os.listdir(root):
        model_path = os.path.join(root, model)
        if not os.path.isdir(model_path):
            continue

        data[model] = {}

        for env in os.listdir(model_path):
            env_path = os.path.join(model_path, env)
            if not os.path.isdir(env_path):
                continue

            runs = []

            for seed in os.listdir(env_path):
                seed_path = os.path.join(env_path, seed)
                if not os.path.isdir(seed_path):
                    continue

                # 🔥 find the file dynamically
                rate_map_file = None
                for f in os.listdir(seed_path):
                    if f.startswith("rate_maps_units_seed"):
                        rate_map_file = os.path.join(seed_path, f)
                        break

                if rate_map_file is None:
                    continue

                with open(rate_map_file, "rb") as f:
                    rate_maps = pickle.load(f)

                runs.append(rate_maps)

            data[model][env] = runs

    return data