import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# BOOTSTRAP
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
# LOAD DATA
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

                episode_files = [
                    os.path.join(seed_path, f)
                    for f in os.listdir(seed_path)
                    if "all_env_episodes_info_seed" in f
                ]

                for episode_file in episode_files:
                    with open(episode_file, "rb") as f:
                        episodes = pickle.load(f)
                    runs.append(episodes)

            if runs:
                data[model][env] = runs

    return data


# --------------------------------------------------
# EPISODE → TIMESTEP EXPANSION
# --------------------------------------------------
def build_timestep_sequences(runs):
    all_sequences = []

    for run in runs:
        seq = []

        for duration in run[-1]["duration"]:
            success = 1 if duration < 15 else 0
            seq.extend([success] * int(duration))

        all_sequences.append(seq)

    return all_sequences


# --------------------------------------------------
# PLOTTING PER MODEL (inside one env)
# --------------------------------------------------
def plot_learning_curve_bootstrap(runs, label=None):

    sequences = build_timestep_sequences(runs)

    min_len = min(len(s) for s in sequences)
    data = np.array([s[:min_len] for s in sequences])

    T = data.shape[1]

    mean = data.mean(axis=0)

    ci_low = np.zeros(T)
    ci_high = np.zeros(T)

    for t in range(T):
        lo, hi = bootstrap_ci(data[:, t])
        ci_low[t] = lo
        ci_high[t] = hi

    x = np.arange(T)

    ax = plt.gca()

    ax.plot(x, mean, label=label)
    ax.fill_between(x, ci_low, ci_high, alpha=0.3)

    # styling
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# --------------------------------------------------
# MAIN: ONE FIGURE PER ENVIRONMENT
# --------------------------------------------------
def plot_all_experiments(root="data"):
    data = load_episode_histories(root)

    # assume same envs across models
    envs = set()
    for model in data:
        envs.update(data[model].keys())

    for env in sorted(envs):

        plt.figure(figsize=(8, 5))

        for model in data:
            if env not in data[model]:
                continue

            runs = data[model][env]
            label = model.split("_")[0]

            plot_learning_curve_bootstrap(runs, label=label)

        plt.xlabel("Timesteps")
        plt.ylabel("Success rate")
        plt.title(f"Learning Curve — {env}")
        # plt.legend()
        plt.tight_layout()

        # save per environment
        save_dir = os.path.join("figures", "env_plots")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{env}_learning_curve.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    plot_all_experiments(root="data")