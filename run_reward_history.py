import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# BOOTSTRAP FUNCTION (yours)
# --------------------------------------------------
def bootstrap_ci(data, n_boot=1000, ci=95):
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lo, hi


# --------------------------------------------------
# LOAD EPISODES
# --------------------------------------------------
def load_episode_histories(root="data"):
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

                # Find all matching episode files
                episode_files = [
                    os.path.join(seed_path, f)
                    for f in os.listdir(seed_path)
                    if "all_env_episodes_info_seed" in f
                ]

                # Load each file found
                for episode_file in episode_files:
                    with open(episode_file, "rb") as f:
                        episodes = pickle.load(f)
                    runs.append(episodes)

            if runs:
                data[model][env] = runs

    return data


# --------------------------------------------------
# PLOT WITHOUT SMOOTHING
# --------------------------------------------------
def plot_learning_curve_bootstrap(runs, label=None):
    all_durations = [run['duration'] for run in runs]

    # align runs
    min_len = min(len(d) for d in all_durations)
    durations = np.array([d[:min_len] for d in all_durations])  # (n_runs, T)

    n_runs, T = durations.shape

    mean = durations.mean(axis=0)

    ci_low = np.zeros(T)
    ci_high = np.zeros(T)

    # bootstrap per timestep
    for t in range(T):
        lo, hi = bootstrap_ci(durations[:, t])
        ci_low[t] = lo
        ci_high[t] = hi

    episodes = np.arange(T)

    plt.plot(episodes, mean, label=label)
    plt.fill_between(episodes, ci_low, ci_high, alpha=0.3)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def plot_all_experiments(root="data"):
    data = load_episode_histories(root)

    plt.figure(figsize=(8, 5))

    for model in data:
        for env in data[model]:
            runs = data[model][env]
            label = f"{model}-{env}"
            plot_learning_curve_bootstrap(runs, label=label)

    plt.xlabel("Episodes")
    plt.ylabel("Duration")
    plt.title("Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# RUN
# --------------------------------------------------

plot_all_experiments(root="data")