"""
FTA architecture:
  Linear(50->20) -> LayerNorm(20, non-adaptive) -> FTA -> Linear(220->1) 

Baseline architecture:
  Linear(50->20) -> LayerNorm(20, non-adaptive) -> ReLU1 -> Linear(20->1) 
"""
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
from activation_recorder import ActivationRecorder
from viz import plot_sparsity_map, display_reward_patch
from networks import Backbone, VxVyGaussianHead

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
N_EPISODES = 500


def record_activations(net, modules_dict, env, ag, placecells, timesteps=2000):
    net.eval()
    rec = ActivationRecorder()
    for name, mod in modules_dict.items():
        rec.attach(mod, name=name)
    start_t = env.t
    while env.t < timesteps + start_t:
        obs, rr, term, _, info = env.step1()
        if env.t - env.episodes['start'][-1] > 15:
            env.reset(episode_meta_info='timeout')
        elif term:
            env.reset(episode_meta_info='completed')
        rec.set_observation(obs)
        state = torch.tensor(placecells.get_state(), dtype=torch.float).t()
        with torch.inference_mode():
            _ = net(state)
        for c in [placecells]: c.update()
    results = {name: rec.get(name) for name in modules_dict}
    rec.detach_all()
    net.train()
    return results


def compute_per_feature(fta_acts, total_tiles, n_features):
    """Sum over tiles per feature from FTA output."""
    per_feature = {}
    for feat_idx in range(n_features):
        sp = {}
        for obs_key, tensors in fta_acts.items():
            vals = []
            for t in tensors:
                arr = t.numpy().reshape(-1, total_tiles, n_features)
                feat_sum = arr[0, :, feat_idx].sum()
                vals.append(float(feat_sum))
            sp[obs_key] = sum(vals) / len(vals)
        per_feature[feat_idx] = sp
    return per_feature


def compute_per_unit(acts, n_units):
    """One map per unit in a flat activation vector."""
    per_unit = {}
    for idx in range(n_units):
        sp = {}
        for obs_key, tensors in acts.items():
            vals = []
            for t in tensors:
                arr = t.numpy().flatten()
                vals.append(float(arr[idx]))
            sp[obs_key] = sum(vals) / len(vals)
        per_unit[idx] = sp
    return per_unit


def save_heatmaps_pdf(per_map, n_maps, env, cfg, pdf_path, title_fn):
    with PdfPages(pdf_path) as pdf:
        for idx in range(n_maps):
            sp = per_map[idx]
            title = title_fn(idx, n_maps)
            fig, ax = plot_sparsity_map(env, sp, bins=40, title=title)
            display_reward_patch(fig, ax, cfg.goal_pos, cfg.goal_radius)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f'Saved {n_maps} heatmaps -> {pdf_path}')


# ══════════════════════════════════════════════════════════════════════════
# FTA agent
# ══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("FTA agent")
print("=" * 60)

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
pc_f = PlaceCells(ag_f, params={'n': N_PLACE_CELLS})

opt_fn = lambda p: torch.optim.SGD(p, lr=ETA, maximize=True)
actor_f = Actor(ag_f, params={'input_layers': [pc_f], 'NeuralNetworkModule': actor_fta_nn,
                              'tau': cfg_fta.tau, 'tau_z': cfg_fta.tau_e, 'optimizer': opt_fn})
critic_f = Critic(ag_f, params={'input_layers': [pc_f], 'NeuralNetworkModule': critic_fta,
                                'tau': cfg_fta.tau, 'tau_z': cfg_fta.tau_e, 'optimizer': opt_fn})

# Train
print('\nTraining (FTA)...')
try:
    for i in (pbar := tqdm(range(cfg_fta.n_episodes), desc=cfg_fta.label)):
        _run_single_episode(env_f, ag_f, actor_f, critic_f, [pc_f], cfg_fta)
        sf = np.mean(np.array(env_f.episodes['meta_info'][-100:]) == 'completed')
        et = np.mean(env_f.episodes['duration'][-100:])
        pbar.set_description(f'{cfg_fta.label} | success: {sf:.2f}, time: {et:.1f}')
        if sf > 0.99 and i > 10: break
except KeyboardInterrupt:
    print('Interrupted')




# ══════════════════════════════════════════════════════════════════════════
# Baseline agent
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Baseline agent")
print("=" * 60)

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
pc_b = PlaceCells(ag_b, params={'n': N_PLACE_CELLS})

opt_fn_b = lambda p: torch.optim.SGD(p, lr=ETA, maximize=True)
actor_b = Actor(ag_b, params={'input_layers': [pc_b], 'NeuralNetworkModule': actor_base_nn,
                              'tau': cfg_base.tau, 'tau_z': cfg_base.tau_e, 'optimizer': opt_fn_b})
critic_b = Critic(ag_b, params={'input_layers': [pc_b], 'NeuralNetworkModule': critic_base,
                                'tau': cfg_base.tau, 'tau_z': cfg_base.tau_e, 'optimizer': opt_fn_b})



# Train
print('\nTraining (Baseline)...')
try:
    for i in (pbar := tqdm(range(cfg_base.n_episodes), desc=cfg_base.label)):
        _run_single_episode(env_b, ag_b, actor_b, critic_b, [pc_b], cfg_base)
        sf = np.mean(np.array(env_b.episodes['meta_info'][-100:]) == 'completed')
        et = np.mean(env_b.episodes['duration'][-100:])
        pbar.set_description(f'{cfg_base.label} | success: {sf:.2f}, time: {et:.1f}')
        if sf > 0.99 and i > 10: break
except KeyboardInterrupt:
    print('Interrupted')



print('\nDone!')
