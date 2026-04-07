from utils import load_runs_out_arrays
from representation_analysis import compute_sparsity_per_timestep, compute_sparsity_per_episode, compute_bin_counts_per_timestep, compute_dead_neurons
from plotting import plot_neurons_over_time, plot_bin_counts_per_percentage
import os   
import argparse
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUT_DIR, 'data')
FIGURES_DIR = os.path.join(OUT_DIR, 'figures')

# ══════════════════════════════════════════════════════════════════════════
# Experiment
# ══════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument("--env_shape", type=str, default="empty")
parser.add_argument("--n_bins", type=int, default=11)
parser.add_argument("--model", type=str, default="FTA")

args = parser.parse_args()

runs_out_arrays = load_runs_out_arrays(os.path.join(DATA_DIR, args.model), args.env_shape)

# Sparsity
sparsity_per_timestep_mean, sparsity_per_timestep_se = compute_sparsity_per_timestep(runs_out_arrays, thres=0.1)
plot_neurons_over_time(x=np.arange(len(sparsity_per_timestep_mean)), y =sparsity_per_timestep_mean, x_label='timesteps', y_label=r'% sparsity', se=sparsity_per_timestep_se, save=True, filename=os.path.join(FIGURES_DIR,args.model, f"env_{args.env_shape}", "sparsity_per_timestep.png"))
sparsity_per_episode_mean, sparsity_per_episode_se = compute_sparsity_per_episode(runs_out_arrays, thres=0.1)
plot_neurons_over_time(x=np.arange(len(sparsity_per_episode_mean)), y =sparsity_per_episode_mean, x_label='episodes', y_label=r'% sparsity', se=sparsity_per_episode_se, save=True, filename=os.path.join(FIGURES_DIR,args.model, f"env_{args.env_shape}", "sparsity_per_episode.png"))

# Bin Count
bin_counts = compute_bin_counts_per_timestep(runs_out_arrays, num_bins=args.n_bins, threshold=0.1)
plot_bin_counts_per_percentage(bin_counts, percentages=[1,2,5,7,10,30,50,70,90,100], save=True, filename=os.path.join(FIGURES_DIR,args.model, f"env_{args.env_shape}"))

# Dead Neurons
dead_neurons_mean, dead_neurons_se = compute_dead_neurons(runs_out_arrays)
plot_neurons_over_time(x=np.arange(len(dead_neurons_mean)), y =dead_neurons_mean, x_label='episodes', y_label=r'% dead neurons', se=dead_neurons_se, save=True, filename=os.path.join(FIGURES_DIR,args.model, f"env_{args.env_shape}", "dead_neurons_per_episode.png"))
