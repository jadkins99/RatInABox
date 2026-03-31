"""
FTA architecture:
  Linear(50->20) -> LayerNorm(20, non-adaptive) -> FTA -> Linear(220->1) 

Baseline architecture:
  Linear(50->20) -> LayerNorm(20, non-adaptive) -> ReLU1 -> Linear(20->1) 
"""
from random import random, seed
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mnt', 'RatInABox'))

import importlib
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import ratinabox
from ratinabox.Neurons import PlaceCells

from ac import Actor, Critic
from experiments import ExperimentConfig, _make_env_and_agent, _run_single_episode
from activation_recorder import ActivationRecorderTimestep, find_layer_module
from viz import plot_sparsity_map, display_reward_patch
from networks import Backbone, VxVyGaussianHead
from plotting import plot_bin_counts_per_percentage, plot_dead_neurons_over_time, plot_bin_counts_per_percentage, plot_fta_average_units_rate_map, plot_occupancy_map, plot_rate_maps, plot_fta_rate_maps
from representation_analysis import compute_dead_neurons_per_timestep, compute_dead_neurons_per_episode,compute_bin_counts_per_timestep, compute_fta_rate_map_single, compute_dead_neurons_per_timestep_single, compute_bin_counts_per_timestep_single
from utils import save_data, load_data, load_runs_bins


OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load PyPI FTA
import importlib.metadata as _meta
_fta_dist = _meta.distribution('fuzzy-tiling-activation')
_fta_pkg_dir = str(_fta_dist._path.parent)
_fta_torch_path = os.path.join(_fta_pkg_dir, 'fta', 'torch.py')
spec = importlib.util.spec_from_file_location(
    'fta_pypi_torch', _fta_torch_path)
fta_pypi_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fta_pypi_mod)
PyPiFTA = fta_pypi_mod.FTA

N_TILES = 10
BOUND = 1.0
PRE_FTA_DIM = 20
N_PLACE_CELLS = 50
ETA = 0.002

T_TIMEOUT = 2
N_EPISODES = 5000
GOAL_POS = np.array([0.5, 0.5])
GOAL_RADIUS = 0.1
REWARD = 1 # Reward
REWARD_DURATION = 1 # Reward duration
OBSTACLES = {
    "empty": [],
    "obstacle_near_goal": [{"x_min": 0.30, "x_max": 0.45, "y_min": 0.30, "y_max": 0.45}],
    "obstacle_far_goal":  [{"x_min": 0.85, "x_max": 1.0,  "y_min": 0.85, "y_max": 1.0}],
}

FIGURES_DIR = os.path.join(OUT_DIR, 'figures')
DATA_DIR = os.path.join(OUT_DIR, 'data')


def get_environment(env, shape="obstacle_near_goal"):
    if shape == "obstacle_near_goal":
        env.add_wall([[.30, .30], [.30, .45]])
        env.add_wall([[.30, .45], [.45, .45]])
        env.add_wall([[.45, .45], [.45, .30]])
        env.add_wall([[.45, .30], [.30, .30]])
        resolution = 0.01  # spacing between walls
        for y in np.arange(0.30, 0.45, resolution):
            env.add_wall([[.30, y], [.45, y]])
    elif shape == "obstacle_far_goal":
        env.add_wall([[0.85, 0.85], [0.85, 1.0]])
        env.add_wall([[0.85, 1.0], [1.0, 1.0]])
        env.add_wall([[1.0, 1.0], [1.0, 0.85]])
        env.add_wall([[1.0, 0.85], [0.85, 0.85]])
        resolution = 0.01
        for y in np.arange(0.85, 1.0, resolution):
            env.add_wall([[0.85, y], [1.0, y]])
    else:
        raise ValueError(f"Unknown environment shape: {shape}")
    return env


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

def create_fta_hook(ag, env, bins, thres=0.1):
    """
    Creates a forward hook that records:
    - agent position
    - timestep
    - raw FTA output
    - binned FTA output
    """

    states = []
    time_steps = []
    out_arrays = []
    fta_bins = []


    def hook_fn(module, inputs, output):

        # ---- convert tensors safely ----
        out_arr = output.detach().cpu().numpy()

        # preserve structure if multi-dim
        out_flat = out_arr.reshape(-1)

        # split into bins (feature groups)
        num_groups = out_flat.size // bins if bins > 0 else 1
        grouped = np.array_split(out_flat, num_groups)

        # ---- record ----
        states.append(np.copy(ag.pos))
        time_steps.append(env.t)
        out_arrays.append(out_flat)
        fta_bins.append(grouped)

    return hook_fn, states, time_steps, fta_bins, out_arrays


def run_multiple_episodes(
    n_episodes,
    env,
    ag,
    actor,
    critic,
    placecells,
    num_bins,
    layer,
    time_limit=15
):
    """
    Runs multiple episodes, records FTA sparsity at each timestep.

    Returns:
        all_episode_sparsity: list of lists of sparsity values (one per episode)
        all_episode_time: list of lists of time steps (one per episode)
    """
    
    
    all_episode_time = []
    all_episodes_state = []
    all_episode_bins = []
    all_out_arrays = []
    

    try: 
        for i in (pbar := tqdm(range(n_episodes), desc = "")):
            # Reset environment and agent
            env.reset()

            # Create new FTA hook for this episode
            hook_fn, states, time_steps, bins, out_arrays = create_fta_hook(layer, ag, env, bins=num_bins)
            handle = layer.register_forward_hook(hook_fn)

            
            # Run one episode
            _run_single_episode(env, ag, actor, critic, state_cells=[placecells], time_limit=time_limit, seed=i)

            # Remove hook
            handle.remove()

            # Store episode data
            
            all_episode_time.append(time_steps)
            all_episodes_state.append(states)
            all_episode_bins.append(bins)
            all_out_arrays.append(out_arrays)

            
            success_frac = np.mean(np.array(env.episodes['meta_info'][-100:]) == "completed")
            episode_time = np.mean(env.episodes['duration'][-100:])
            pbar.set_description(f"<success fraction>: {success_frac:.2f}, <episode time> {episode_time:.1f}")
            if success_frac > 0.99 and i > 10:
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    
    return all_episode_time, all_episodes_state, all_episode_bins, all_out_arrays

def run_experiment(env,ag, placecells,actor,critic,layer, env_shape="empty"):

    plot_rate_maps(env, ag, placecells, actor, critic, GOAL_POS, GOAL_RADIUS,time = 'before', save_dir=os.path.join(FIGURES_DIR,f"env_{env_shape}", f"seed_{seed}"))
    fta = find_layer_module(critic.NeuralNetworkModule, layer)
    all_episodes_time, all_episodes_state, all_episodes_bins, all_out_arrays = run_multiple_episodes(n_episodes=N_EPISODES,env=env,ag=ag,actor=actor,critic=critic,placecells=placecells,fta=fta,num_bins=N_TILES,time_limit=T_TIMEOUT)

    save_data(all_episodes_time, os.path.join(DATA_DIR, f"env_{env_shape}", f"seed_{seed}", f'all_episodes_time_seed_{seed}'))
    save_data(all_episodes_state, os.path.join(DATA_DIR, f"env_{env_shape}", f"seed_{seed}", f'all_episodes_states_seed_{seed}'))
    save_data(all_episodes_bins, os.path.join(DATA_DIR, f"env_{env_shape}",f"seed_{seed}", f'all_episodes_bins_seed_{seed}'))
    save_data(all_out_arrays, os.path.join(DATA_DIR,f"env_{env_shape}",f"seed_{seed}", f'all_out_arrays_seed_{seed}'))

    plot_rate_maps(env, ag, placecells, actor, critic, GOAL_POS, GOAL_RADIUS,time = 'after', reward=True, trajectory=True, save_dir=os.path.join(FIGURES_DIR,f"env_{env_shape}", f"seed_{seed}"))

    rate_maps, occupancy = compute_fta_rate_map_single(all_episodes_state, all_out_arrays, n_bins=N_TILES, filter_size=1.5)
    plot_fta_rate_maps(rate_maps, n_cols=N_TILES, save_dir=os.path.join(FIGURES_DIR, f"env_{env_shape}", f"seed_{seed}"), filename=f"fta_rate_maps_seed_{seed}.png")
    plot_occupancy_map(occupancy, save_dir=os.path.join(FIGURES_DIR, f"env_{env_shape}", f"seed_{seed}"), filename=f"occupancy_seed_{seed}.png")

    rate_maps_obs, occupancy_obs = compute_fta_rate_map_single(all_episodes_state, all_out_arrays, n_bins=N_TILES, filter_size=1.5, obstacles=OBSTACLES[env_shape])
    plot_fta_rate_maps(rate_maps_obs, n_cols=N_TILES, save_dir=os.path.join(FIGURES_DIR, f"env_{env_shape}", f"seed_{seed}"), filename=f"fta_rate_maps_obs_seed_{seed}.png")
    plot_occupancy_map(occupancy_obs, save_dir=os.path.join(FIGURES_DIR, f"env_{env_shape}", f"seed_{seed}"), filename=f"occupancy_obs_seed_{seed}.png")

    plot_fta_average_units_rate_map(rate_maps, save_dir=os.path.join(FIGURES_DIR, f"env_{env_shape}", f"seed_{seed}"), filename=f"fta_rate_map_avg_units_seed_{seed}.png")
    plot_fta_average_units_rate_map(rate_maps_obs, save_dir=os.path.join(FIGURES_DIR, f"env_{env_shape}", f"seed_{seed}"), filename=f"fta_rate_map_avg_units_obs_seed_{seed}.png")

    #Dead neurons
    dead_neurons = compute_dead_neurons_per_timestep_single(all_out_arrays)
    plot_dead_neurons_over_time(x=np.arange(len(dead_neurons)), y =dead_neurons, x_label='timesteps', y_label='%\ dead neurons', save=True, filename=os.path.join(FIGURES_DIR, f"env_{env_shape}", f"seed_{seed}", "dead_neurons_per_timestep.png"))
    bin_count = compute_bin_counts_per_timestep_single(all_out_arrays, n_bins=N_TILES)
    plot_bin_counts_per_percentage(bin_count, percentages=[1,2,5,7,10,30,50,70,90,100], save=True, filename=os.path.join(FIGURES_DIR, f"env_{env_shape}",f"seed_{seed}"))


# ══════════════════════════════════════════════════════════════════════════
# Experiment
# ══════════════════════════════════════════════════════════════════════════

 # parser = argparse.ArgumentParser()
    # parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--env_shape", type=str, default="empty")
    # args = parser.parse_args()

# ══════════════════════════════════════════════════════════════════════════
# FTA agent
# ══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("FTA agent")
print("=" * 60)


set_seed(seed)

print(f"Starting experiment (seed {seed}") 

pypi_fta = PyPiFTA(
    bound=BOUND, spillover_base=0, spillover_mode='derive_from_tile_width',
    tile_width=None, num_tiles=N_TILES,
)
total_tiles = pypi_fta.num_tiles  # 11
fta_out_dim = PRE_FTA_DIM * total_tiles  # 220

post_fta_relu = nn.ReLU()

critic_fta = nn.Sequential(
    nn.Linear(N_PLACE_CELLS, PRE_FTA_DIM),              # 0
    nn.LayerNorm(PRE_FTA_DIM, elementwise_affine=False), # 1: non-adaptive
    pypi_fta,                                             # 2
    nn.Linear(fta_out_dim, 1),                 # 3: compress 220 -> 1
    # post_fta_relu,                                        # 4
    # nn.Linear(PRE_FTA_DIM, 1),                            # 5
)
print(f'\n{critic_fta}')

actor_fta_nn = VxVyGaussianHead(Backbone(n_in=N_PLACE_CELLS, n_out=2, hidden=[50]))

cfg_fta = ExperimentConfig(label='FTA_representation', n_episodes=N_EPISODES, eta=ETA)
env_f, ag_f = _make_env_and_agent(cfg_fta)
env_f = get_environment(env_f, shape="obstacle_near_goal")
pc_f = PlaceCells(ag_f, params={'n': N_PLACE_CELLS})

opt_fn = lambda p: torch.optim.SGD(p, lr=ETA, maximize=True)
actor_f = Actor(ag_f, params={'input_layers': [pc_f], 'NeuralNetworkModule': actor_fta_nn,
                              'tau': cfg_fta.tau, 'tau_z': cfg_fta.tau_e, 'optimizer': opt_fn})
critic_f = Critic(ag_f, params={'input_layers': [pc_f], 'NeuralNetworkModule': critic_fta,
                                'tau': cfg_fta.tau, 'tau_z': cfg_fta.tau_e, 'optimizer': opt_fn})

print("\nRecording FTA activations per timestep...")






# ══════════════════════════════════════════════════════════════════════════
# Baseline agent
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Baseline agent")
print("=" * 60)

set_seed(seed)

print(f"Starting experiment (seed {seed}") 

baseline_relu1 = nn.ReLU()
# baseline_relu2 = nn.ReLU()

critic_base = nn.Sequential(
    nn.Linear(N_PLACE_CELLS, PRE_FTA_DIM),  # 0
    nn.LayerNorm(PRE_FTA_DIM, elementwise_affine=False), # 1: non-adaptive
    baseline_relu1,                           # 1
    nn.Linear(PRE_FTA_DIM, 1),     # 2
    # baseline_relu2,                           # 3
    # nn.Linear(PRE_FTA_DIM, 1),               # 4
)
print(f'\n{critic_base}')

actor_base_nn = VxVyGaussianHead(Backbone(n_in=N_PLACE_CELLS, n_out=2, hidden=[50]))

cfg_base = ExperimentConfig(label='Baseline', n_episodes=N_EPISODES, eta=ETA)
env_b, ag_b = _make_env_and_agent(cfg_base)
env_b = get_environment(env_b, shape="obstacle_near_goal")
pc_b = PlaceCells(ag_b, params={'n': N_PLACE_CELLS})

opt_fn_b = lambda p: torch.optim.SGD(p, lr=ETA, maximize=True)
actor_b = Actor(ag_b, params={'input_layers': [pc_b], 'NeuralNetworkModule': actor_base_nn,
                              'tau': cfg_base.tau, 'tau_z': cfg_base.tau_e, 'optimizer': opt_fn_b})
critic_b = Critic(ag_b, params={'input_layers': [pc_b], 'NeuralNetworkModule': critic_base,
                                'tau': cfg_base.tau, 'tau_z': cfg_base.tau_e, 'optimizer': opt_fn_b})



print("\nRecording Baseline activations per timestep...")





print('\nDone!')
