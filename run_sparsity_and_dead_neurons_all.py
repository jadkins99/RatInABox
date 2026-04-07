from utils import load_runs_out_arrays
from representation_analysis import compute_sparsity_per_episode, compute_dead_neurons
from plotting import plot_multiple_models
import argparse
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUT_DIR, 'data')
FIGURES_DIR = os.path.join(OUT_DIR, 'figures')

# os.makedirs(FIGURES_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# Experiment
# ══════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument("--env_shape", type=str, default="empty")

args = parser.parse_args()

models = ["FTA", "ReLU_20_units", "ReLU_220_units"]  

colors = {
    "FTA": "blue",
    "ReLU_20_units": "orange",
    "ReLU_220_units": "green",
}

sparsity_results = {}
dead_results = {}

# ══════════════════════════════════════════════════════════════════════════
# Compute metrics
# ══════════════════════════════════════════════════════════════════════════

for model in models:
    print(f"\nProcessing model: {model}")

    runs_out_arrays = load_runs_out_arrays(
        os.path.join(DATA_DIR, model),
        args.env_shape
    )

    # Sparsity per timestep
    sp_mean, sp_se = compute_sparsity_per_episode(
        runs_out_arrays, thres=0.1
    )
    sparsity_results[model] = (sp_mean, sp_se)

    # Dead neurons per episode
    dn_mean, dn_se = compute_dead_neurons(runs_out_arrays)
    dead_results[model] = (dn_mean, dn_se)


# ══════════════════════════════════════════════════════════════════════════
# Plot: Sparsity
# ══════════════════════════════════════════════════════════════════════════

plot_multiple_models(
    sparsity_results,
    x_label="episodes",
    y_label="% sparsity",
    save=True,
    filename=os.path.join(
        FIGURES_DIR,
        f"sparsity_per_episode_all_models_env_{args.env_shape}.png"
    )
)

# ══════════════════════════════════════════════════════════════════════════
# Plot: Dead neurons
# ══════════════════════════════════════════════════════════════════════════

plot_multiple_models(
    dead_results,
    x_label="episodes",
    y_label="% dead neurons",
    save=True,
    filename=os.path.join(
        FIGURES_DIR,
        f"dead_neurons_all_models_env_{args.env_shape}.png"
    )
)