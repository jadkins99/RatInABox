"""
FTA architecture:
  Linear(50->20) -> LayerNorm(20, non-adaptive) -> FTA -> Linear(220->1) 

Baseline architecture:
  Linear(50->20) -> LayerNorm(20, non-adaptive) -> ReLU1 -> Linear(20->1) 
"""
import argparse
import random
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
from activation_recorder import find_penultimate_layer
from viz import plot_sparsity_map, display_reward_patch
from networks import Backbone, VxVyGaussianHead
from plotting import plot_average_units_rate_map, plot_bin_counts_per_percentage, plot_neurons_over_time, plot_bin_counts_per_percentage, plot_occupancy_map, plot_rate_maps, plot_units_rate_maps
from representation_analysis import compute_sparsity_per_timestep_single, compute_bin_counts_per_timestep_single, compute_rate_maps_single, compute_sparsity_per_episode_single
from utils import save_data


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

N_EPISODES = 3

OBSTACLES = {
    "empty": [],

    "obstacle_near_goal": [
        {"x_min": 0.30, "x_max": 0.45, "y_min": 0.30, "y_max": 0.45}
    ],

    "obstacle_corners": [
        # bottom-left
        {"x_min": 0.0,  "x_max": 0.15, "y_min": 0.0,  "y_max": 0.15},

        # bottom-right
        {"x_min": 0.85, "x_max": 1.0,  "y_min": 0.0,  "y_max": 0.15},

        # top-left
        {"x_min": 0.0,  "x_max": 0.15, "y_min": 0.85, "y_max": 1.0},

        # top-right
        {"x_min": 0.85, "x_max": 1.0,  "y_min": 0.85, "y_max": 1.0},
    ],

    # NEW: walls
    "walls": [
        # vertical wall: x = 0.40 from y = 0 → 0.60
        {"x_min": 0.40, "x_max": 0.40, "y_min": 0.0,  "y_max": 0.60},

        # horizontal wall: y = 0.60 from x = 0.40 → 0.80
        {"x_min": 0.40, "x_max": 0.80, "y_min": 0.60, "y_max": 0.60},
    ]
}

FIGURES_DIR = os.path.join(OUT_DIR, 'figures')
DATA_DIR = os.path.join(OUT_DIR, 'data')


def get_environment(env, shape="empty"):
    """
    Adds obstacles to the environment based on OBSTACLES dict.
    Each obstacle is defined by x_min, x_max, y_min, y_max.
    """

    obstacles = OBSTACLES.get(shape, [])

    resolution = 0.01  # spacing for filling obstacles

    for obs in obstacles:
        x_min, x_max = obs["x_min"], obs["x_max"]
        y_min, y_max = obs["y_min"], obs["y_max"]

        # ---- outer walls (rectangle boundary) ----
        env.add_wall([[x_min, y_min], [x_min, y_max]])  # left
        env.add_wall([[x_min, y_max], [x_max, y_max]])  # top
        env.add_wall([[x_max, y_max], [x_max, y_min]])  # right
        env.add_wall([[x_max, y_min], [x_min, y_min]])  # bottom

        # ---- fill interior with horizontal lines ----
        for y in np.arange(y_min, y_max, resolution):
            env.add_wall([[x_min, y], [x_max, y]])

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

def create_hook(ag, env, thres=0.1):
    """
    Creates a forward hook that records:
    - agent position
    - timestep
    - raw output
    """

    states = []
    time_steps = []
    out_arrays = []


    def hook_fn(module, inputs, output):

        # ---- convert tensors safely ----
        out_arr = output.detach().cpu().numpy()

        # preserve structure if multi-dim
        out_flat = out_arr.reshape(-1)

        # ---- record ----
        states.append(np.copy(ag.pos))
        time_steps.append(env.t)
        out_arrays.append(out_flat)
        

    return hook_fn, states, time_steps, out_arrays


def run_multiple_episodes(
    env,
    ag,
    actor,
    critic,
    placecells,
    num_bins,
    layer,
    experiment_cfg
):
    """
    Runs multiple episodes, records FTA sparsity at each timestep.

    Returns:
        all_episode_sparsity: list of lists of sparsity values (one per episode)
        all_episode_time: list of lists of time steps (one per episode)
    """
    
    
    all_episode_time = []
    all_episodes_state = []
    all_out_arrays = []

    try: 
        for i in (pbar := tqdm(range(experiment_cfg.n_episodes), desc = "")):
            # Reset environment and agent
            env.reset()

            # Create new FTA hook for this episode
            hook_fn, states, time_steps, out_arrays = create_hook(ag, env)
            handle = layer.register_forward_hook(hook_fn)

            
            # Run one episode
            _run_single_episode(env, ag, actor, critic, state_cells=[placecells], cfg=experiment_cfg)

            # Remove hook
            handle.remove()

            # Store episode data
            
            all_episode_time.append(time_steps)
            all_episodes_state.append(states)
            all_out_arrays.append(out_arrays)

            success_frac = np.mean(np.array(env.episodes['meta_info'][-100:]) == "completed")
            episode_time = np.mean(env.episodes['duration'][-100:])
            pbar.set_description(f"<success fraction>: {success_frac:.2f}, <episode time> {episode_time:.1f}")
            if success_frac > 0.99 and i > 10:
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    
    return all_episode_time, all_episodes_state, all_out_arrays

def run_experiment(env,ag, placecells,actor,critic,n_bins,experiment_cfg, env_shape="empty", seed=0):

    layer = find_penultimate_layer(critic.NeuralNetworkModule)

    model = experiment_cfg.label

    print(f"Running experiment for model {model} in env {env_shape} with seed {seed}...")

    print(f"Plotting initial rate maps...")
    plot_rate_maps(env, ag, placecells, actor, critic, experiment_cfg.goal_pos, experiment_cfg.goal_radius, time='before', save_dir=os.path.join(FIGURES_DIR, model,f"env_{env_shape}", f"seed_{seed}"))
    
    all_episodes_time, all_episodes_state, all_out_arrays = run_multiple_episodes(env=env,ag=ag,actor=actor,critic=critic,placecells=placecells,num_bins=n_bins,layer=layer,experiment_cfg=experiment_cfg)

    print(f"Experiment completed. Saving data and plotting results...")
    save_data(all_episodes_time, os.path.join(DATA_DIR,model, f"env_{env_shape}", f"seed_{seed}", f'all_episodes_time_seed_{seed}'))
    save_data(all_episodes_state, os.path.join(DATA_DIR,model, f"env_{env_shape}", f"seed_{seed}", f'all_episodes_states_seed_{seed}'))
    save_data(all_out_arrays, os.path.join(DATA_DIR,model, f"env_{env_shape}",f"seed_{seed}", f'all_out_arrays_seed_{seed}'))

    print(f"Plotting rate maps after experiment...")
    plot_rate_maps(env, ag, placecells, actor, critic, experiment_cfg.goal_pos, experiment_cfg.goal_radius,time = 'after', reward=True, trajectory=True, save_dir=os.path.join(FIGURES_DIR, model,f"env_{env_shape}", f"seed_{seed}"))

    print(f"Computing and plotting unit rate maps and occupancy maps...")
    rate_maps, occupancy = compute_rate_maps_single(all_episodes_state, all_out_arrays, filter_size=1.5, obstacles=OBSTACLES[env_shape])
    plot_units_rate_maps(rate_maps, save_dir=os.path.join(FIGURES_DIR,model, f"env_{env_shape}", f"seed_{seed}"), filename=f"{model.lower()}_rate_maps_seed_{seed}.png")
    plot_occupancy_map(occupancy, save_dir=os.path.join(FIGURES_DIR, model, f"env_{env_shape}", f"seed_{seed}"), filename=f"{model.lower()}_occupancy_seed_{seed}.png")

    print(f"Plotting average unit rate maps...")
    plot_average_units_rate_map(rate_maps, save_dir=os.path.join(FIGURES_DIR, model, f"env_{env_shape}", f"seed_{seed}"), filename=f"{model.lower()}_rate_map_avg_units_seed_{seed}_env_{env_shape}.png")

    #Dead neurons
    print(f"Computing and plotting sparsity over time and bin counts...")
    sparsity_timestep = compute_sparsity_per_timestep_single(all_out_arrays)
    plot_neurons_over_time(x=np.arange(len(sparsity_timestep)), y =sparsity_timestep, x_label='timesteps', y_label=r'% sparsity', save=True, filename=os.path.join(FIGURES_DIR, model, f"env_{env_shape}", f"seed_{seed}", "sparsity_per_timestep.png"))
    sparsity_episode = compute_sparsity_per_episode_single(all_out_arrays)
    plot_neurons_over_time(x=np.arange(len(sparsity_episode)), y =sparsity_episode, x_label='episodes', y_label=r'% sparsity', save=True, filename=os.path.join(FIGURES_DIR, model, f"env_{env_shape}", f"seed_{seed}", "sparsity_per_episode.png"))

    
    # bin_count = compute_bin_counts_per_timestep_single(all_out_arrays, num_bins=n_bins)
    # plot_bin_counts_per_percentage(bin_count, percentages=[1,2,5,7,10,30,50,70,90,100], save=True, filename=os.path.join(FIGURES_DIR, model, f"env_{env_shape}",f"seed_{seed}"))


# ══════════════════════════════════════════════════════════════════════════
# Experiment
# ══════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--env_shape", type=str, default="empty")
args = parser.parse_args()

# ══════════════════════════════════════════════════════════════════════════
# FTA - Layer agent
# ══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("FTA agent")
print("=" * 60)

set_seed(args.seed)

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

cfg_fta = ExperimentConfig(label='FTA', n_episodes=N_EPISODES, eta=ETA)
env_f, ag_f = _make_env_and_agent(cfg_fta)
env_f = get_environment(env_f, shape=args.env_shape)
pc_f = PlaceCells(ag_f, params={'n': N_PLACE_CELLS})

opt_fn = lambda p: torch.optim.SGD(p, lr=ETA, maximize=True)
actor_f = Actor(ag_f, params={'n':2,'input_layers': [pc_f], 'NeuralNetworkModule': actor_fta_nn,
                              'tau': cfg_fta.tau, 'tau_z': cfg_fta.tau_e, 'optimizer': opt_fn})
critic_f = Critic(ag_f, params={'n':1,'input_layers': [pc_f], 'NeuralNetworkModule': critic_fta,
                                'tau': cfg_fta.tau, 'tau_z': cfg_fta.tau_e, 'optimizer': opt_fn})

print(f"Starting experiment") 

run_experiment(env_f, ag_f, pc_f, actor_f, critic_f, n_bins=total_tiles, env_shape=args.env_shape, experiment_cfg=cfg_fta, seed = args.seed,)

# ══════════════════════════════════════════════════════════════════════════
# ReLU FTA - Layer agent
# ══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("ReLU FTA agent")
print("=" * 60)

set_seed(args.seed)

pypi_relu_fta = PyPiFTA(
    bound=BOUND, spillover_base=0, spillover_mode='derive_from_tile_width',
    tile_width=None, num_tiles=N_TILES,
)
total_tiles = pypi_relu_fta.num_tiles  # 11
fta_out_dim = PRE_FTA_DIM * total_tiles  # 220

pre_fta_relu = nn.ReLU()

critic_relu_fta = nn.Sequential(
    nn.Linear(N_PLACE_CELLS, PRE_FTA_DIM),              # 0
    nn.LayerNorm(PRE_FTA_DIM, elementwise_affine=False), # 1: non-adaptive
    pre_fta_relu,
    nn.Linear(PRE_FTA_DIM, PRE_FTA_DIM),              # 2: expand to 20 -> 20
    pypi_relu_fta,                                             # 2
    nn.Linear(fta_out_dim, 1),                 # 3: compress 220 -> 1                          
)
print(f'\n{critic_relu_fta}')

actor_relu_fta_nn = VxVyGaussianHead(Backbone(n_in=N_PLACE_CELLS, n_out=2, hidden=[50]))

cfg_relu_fta = ExperimentConfig(label='ReLU_FTA', n_episodes=N_EPISODES, eta=ETA)
env_relu_fta, ag_relu_fta = _make_env_and_agent(cfg_relu_fta)
env_relu_fta = get_environment(env_relu_fta, shape=args.env_shape)
pc_relu_fta = PlaceCells(ag_relu_fta, params={'n': N_PLACE_CELLS})

opt_fn = lambda p: torch.optim.SGD(p, lr=ETA, maximize=True)
actor_relu_fta = Actor(ag_relu_fta, params={'n':2,'input_layers': [pc_relu_fta], 'NeuralNetworkModule': actor_relu_fta_nn,
                              'tau': cfg_relu_fta.tau, 'tau_z': cfg_relu_fta.tau_e, 'optimizer': opt_fn})
critic_relu_fta = Critic(ag_relu_fta, params={'n':1,'input_layers': [pc_relu_fta], 'NeuralNetworkModule': critic_relu_fta,
                                'tau': cfg_relu_fta.tau, 'tau_z': cfg_relu_fta.tau_e, 'optimizer': opt_fn})

print(f"Starting experiment") 

run_experiment(env_relu_fta, ag_relu_fta, pc_relu_fta, actor_relu_fta, critic_relu_fta, n_bins=total_tiles, env_shape=args.env_shape, experiment_cfg=cfg_relu_fta, seed = args.seed,)


# ══════════════════════════════════════════════════════════════════════════
# Baseline agent ReLU 20 units
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Baseline agent ReLU 20 units")
print("=" * 60)

set_seed(args.seed)

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

cfg_base = ExperimentConfig(label='ReLU_20_units', n_episodes=N_EPISODES, eta=ETA)
env_b, ag_b = _make_env_and_agent(cfg_base)
env_b = get_environment(env_b, shape=args.env_shape)
pc_b = PlaceCells(ag_b, params={'n': N_PLACE_CELLS})

opt_fn_b = lambda p: torch.optim.SGD(p, lr=ETA, maximize=True)
actor_b = Actor(ag_b, params={'n':2,'input_layers': [pc_b], 'NeuralNetworkModule': actor_base_nn,
                              'tau': cfg_base.tau, 'tau_z': cfg_base.tau_e, 'optimizer': opt_fn_b})
critic_b = Critic(ag_b, params={'n':1,'input_layers': [pc_b], 'NeuralNetworkModule': critic_base,
                                'tau': cfg_base.tau, 'tau_z': cfg_base.tau_e, 'optimizer': opt_fn_b})



print(f"Starting experiment") 

run_experiment(env_b, ag_b, pc_b, actor_b, critic_b, n_bins=20, experiment_cfg=cfg_base, env_shape=args.env_shape, seed = args.seed)

# ══════════════════════════════════════════════════════════════════════════
# Baseline agent ReLU 220 units
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Baseline agent ReLU 220 units")
print("=" * 60)

set_seed(args.seed)

baseline_relu2 = nn.ReLU()
# baseline_relu2 = nn.ReLU()

critic_base = nn.Sequential(
    nn.Linear(N_PLACE_CELLS, 220),  # 0
    nn.LayerNorm(220, elementwise_affine=False), # 1: non-adaptive
    baseline_relu2,                           # 1
    nn.Linear(220, 1),     # 2
    # baseline_relu2,                           # 3
    # nn.Linear(PRE_FTA_DIM, 1),               # 4
)
print(f'\n{critic_base}')

actor_base_220_nn = VxVyGaussianHead(Backbone(n_in=N_PLACE_CELLS, n_out=2, hidden=[50]))

cfg_base_220 = ExperimentConfig(label='ReLU_220_units', n_episodes=N_EPISODES, eta=ETA)
env_b_220, ag_b_220 = _make_env_and_agent(cfg_base_220)
env_b_220 = get_environment(env_b_220, shape=args.env_shape)
pc_b_220 = PlaceCells(ag_b_220, params={'n': N_PLACE_CELLS})

opt_fn_b_220 = lambda p: torch.optim.SGD(p, lr=ETA, maximize=True)
actor_b_220 = Actor(ag_b_220, params={'n':2,'input_layers': [pc_b_220], 'NeuralNetworkModule': actor_base_220_nn,
                              'tau': cfg_base_220.tau, 'tau_z': cfg_base_220.tau_e, 'optimizer': opt_fn_b_220})
critic_b_220 = Critic(ag_b_220, params={'n':1,'input_layers': [pc_b_220], 'NeuralNetworkModule': critic_base,
                                'tau': cfg_base_220.tau, 'tau_z': cfg_base_220.tau_e, 'optimizer': opt_fn_b_220})


print(f"Starting experiment") 

run_experiment(env_b_220, ag_b_220, pc_b_220, actor_b_220, critic_b_220, n_bins=220, experiment_cfg=cfg_base_220, env_shape=args.env_shape, seed = args.seed,)

# ══════════════════════════════════════════════════════════════════════════
# Baseline agent ReLU - ReLU 220 units
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Baseline agent ReLU 220 units")
print("=" * 60)

set_seed(args.seed)



baseline_double_relu_1 = nn.ReLU()
baseline_double_relu_2 = nn.ReLU()
# baseline_relu2 = nn.ReLU()

critic_double_relu = nn.Sequential(
    nn.Linear(N_PLACE_CELLS, 20),  # 0
    nn.LayerNorm(20, elementwise_affine=False), # 1: non-adaptive
    baseline_double_relu_1,                           # 1
    nn.Linear(20, 220),     # 2
    baseline_double_relu_2,                           # 3
    nn.Linear(220, 1),               # 4
)
print(f'\n{critic_double_relu}')

actor_double_relu_220_nn = VxVyGaussianHead(Backbone(n_in=N_PLACE_CELLS, n_out=2, hidden=[50]))

cfg_double_relu_220 = ExperimentConfig(label='Double_ReLU_220_units', n_episodes=N_EPISODES, eta=ETA)
env_double_relu_220, ag_double_relu_220 = _make_env_and_agent(cfg_double_relu_220)
env_double_relu_220 = get_environment(env_double_relu_220, shape=args.env_shape)
pc_double_relu_220 = PlaceCells(ag_double_relu_220, params={'n': N_PLACE_CELLS})

opt_fn_double_220 = lambda p: torch.optim.SGD(p, lr=ETA, maximize=True)
actor_double_relu_220 = Actor(ag_double_relu_220, params={'n':2,'input_layers': [pc_double_relu_220], 'NeuralNetworkModule': actor_double_relu_220_nn,
                              'tau': cfg_double_relu_220.tau, 'tau_z': cfg_double_relu_220.tau_e, 'optimizer': opt_fn_double_220})
critic_double_relu_220 = Critic(ag_double_relu_220, params={'n':1,'input_layers': [pc_double_relu_220], 'NeuralNetworkModule': critic_double_relu,
                                'tau': cfg_double_relu_220.tau, 'tau_z': cfg_double_relu_220.tau_e, 'optimizer': opt_fn_double_220})


print(f"Starting experiment") 

run_experiment(env_double_relu_220, ag_double_relu_220, pc_double_relu_220, actor_double_relu_220, critic_double_relu_220, n_bins=220, experiment_cfg=cfg_double_relu_220, env_shape=args.env_shape, seed = args.seed,)


print('\nDone!')


