from utils import load_runs_out_arrays
from representation_analysis import compute_sparsity_per_episode, compute_dead_neurons
from plotting import plot_multiple_models
import argparse
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUT_DIR, 'data')
FIGURES_DIR = os.path.join(OUT_DIR, 'figures')

parser = argparse.ArgumentParser()
parser.add_argument("--env_shape", type=str, default="empty")
args = parser.parse_args()

models = ["FTA", "ReLU_20_units", "ReLU_220_units"]

sparsity_results = {}
dead_results = {}

# =========================
# Compute metrics (BOOTSTRAP CI)
# =========================

for model in models:
    print(f"\nProcessing model: {model}")

    runs_out_arrays = load_runs_out_arrays(
        os.path.join(DATA_DIR, model),
        args.env_shape
    )

    # -------------------------
    # Sparsity (bootstrap CI)
    # -------------------------
    sp_mean, sp_low, sp_high = compute_sparsity_per_episode(
        runs_out_arrays, thres=0.1
    )

    sparsity_results[model] = (sp_mean, sp_low, sp_high)

    # -------------------------
    # Dead neurons (bootstrap CI)
    # -------------------------
    dn_mean, dn_low, dn_high = compute_dead_neurons(
        runs_out_arrays
    )

    dead_results[model] = (dn_mean, dn_low, dn_high)


# =========================
# Plot: Sparsity
# =========================

print("\nPlotting sparsity for all models...")

for model in models:
    print(f"Model: {model}")
    print(f"  Sparsity mean: {sparsity_results[model][0][-1]:.2f}%")
    print(f"  Sparsity 95% CI: [{sparsity_results[model][1][-1]:.2f}%, {sparsity_results[model][2][-1]:.2f}%]")


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

# =========================
# Plot: Dead neurons
# =========================

print("\nPlotting dead neurons for all models...")

for model in models:
    print(f"Model: {model}")
    print(f"  Dead neurons mean: {dead_results[model][0][-1]:.2f}%")
    print(f"  Dead neurons 95% CI: [{dead_results[model][1][-1]:.2f}%, {dead_results[model][2][-1]:.2f}%]")


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