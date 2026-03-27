"""PyPI FTA: Linear -> LayerNorm -> FTA (no ReLU), before & after learning."""
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

OUT_DIR = os.path.join(os.path.dirname(__file__), 'mnt', 'RatInABox')

# Load PyPI FTA
spec = importlib.util.spec_from_file_location(
    'fta_pypi_torch',
    '/sessions/focused-laughing-newton/.local/lib/python3.10/site-packages/fta/torch.py'
)
fta_pypi_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fta_pypi_mod)
PyPiFTA = fta_pypi_mod.FTA

N_TILES = 10
BOUND = 1.0
TILE_WIDTH = 2 * BOUND / N_TILES  # 0.2 = delta

pypi_fta = PyPiFTA(
    bound=BOUND,
    spillover_base=0,   # eta = delta
    spillover_mode='derive_from_tile_width',
    tile_width=None,
    num_tiles=N_TILES,
)
print(f'PyPI FTA: num_tiles={pypi_fta.num_tiles}, '
      f'tile_width={pypi_fta.tile_width:.3f}, spillover={pypi_fta.spillover}')

pre_fta_dim = 20
fta_out_dim = pre_fta_dim * pypi_fta.num_tiles  # 20 * 11 = 220

class BackboneWrapper(nn.Module):
    def __init__(self, seq):
        super().__init__()
        self.net = seq
    def forward(self, x):
        return self.net(x)

# Linear -> LayerNorm -> FTA (NO ReLU)
critic_nn = BackboneWrapper(nn.Sequential(
    nn.Linear(50, pre_fta_dim),
    nn.LayerNorm(pre_fta_dim),
    pypi_fta,
    nn.Linear(fta_out_dim, 1),
))
print(f'\nCritic architecture:\n{critic_nn}')

actor_nn = VxVyGaussianHead(Backbone(n_in=50, n_out=2, hidden=[50]))

cfg = ExperimentConfig(
    label='PyPI FTA Linear+LN (no ReLU)',
    n_episodes=500,
)
env, ag = _make_env_and_agent(cfg)
pc = PlaceCells(ag, params={'n': 50})

opt_fn = lambda p: torch.optim.SGD(p, lr=cfg.eta, maximize=True)
actor = Actor(ag, params={'input_layers': [pc], 'NeuralNetworkModule': actor_nn,
                          'tau': cfg.tau, 'tau_z': cfg.tau_e, 'optimizer': opt_fn})
critic = Critic(ag, params={'input_layers': [pc], 'NeuralNetworkModule': critic_nn,
                            'tau': cfg.tau, 'tau_z': cfg.tau_e, 'optimizer': opt_fn})

# ── Helpers ──────────────────────────────────────────────────────────────

def record_activations(net, fta_mod, env, ag, placecells, timesteps=2000):
    net.eval()
    rec = ActivationRecorder()
    rec.attach(fta_mod, name='fta')
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
    acts = rec.get('fta')
    rec.detach_all()
    net.train()
    return acts

def compute_pypi_per_tile(fta_acts, total_tiles, pre_dim):
    per_tile = {}
    for tile_idx in range(total_tiles):
        sp = {}
        for obs_key, tensors in fta_acts.items():
            vals = []
            for t in tensors:
                arr = t.numpy().reshape(-1, total_tiles, pre_dim)
                tile_act = arr[0, tile_idx, :]
                above = tile_act[tile_act > 0.01]
                vals.append(float(np.sum(above) / tile_act.size) if tile_act.size > 0 else 0.0)
            sp[obs_key] = sum(vals) / len(vals)
        per_tile[tile_idx] = sp
    return per_tile

def save_heatmaps(per_tile_dict, pdf_path, env, total_tiles, tiling, tw, cfg):
    with PdfPages(pdf_path) as pdf:
        for tile_idx in range(total_tiles):
            sp = per_tile_dict[tile_idx]
            if tile_idx < len(tiling):
                lo, hi = tiling[tile_idx], tiling[tile_idx] + tw
                title = f'Tile {tile_idx+1}/{total_tiles} [{lo:.2f}, {hi:.2f})'
            else:
                title = f'Tile {tile_idx+1}/{total_tiles} [right linear]'
            fig, ax = plot_sparsity_map(env, sp, bins=40, title=title)
            display_reward_patch(fig, ax, cfg.goal_pos, cfg.goal_radius)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f'Saved {total_tiles} heatmaps -> {pdf_path}')

tiling = pypi_fta._tiling.cpu().numpy()
tw = pypi_fta.tile_width
total_tiles = pypi_fta.num_tiles

# ── Before learning ─────────────────────────────────────────────────────

print('\nRecording BEFORE learning...')
acts_before = record_activations(critic_nn, pypi_fta, env, ag, pc)
print(f'Recorded {len(acts_before)} locations')

pt_before = compute_pypi_per_tile(acts_before, total_tiles, pre_fta_dim)
save_heatmaps(pt_before, os.path.join(OUT_DIR, 'per_tile_sparsity_pypi_ln_norelu_before.pdf'),
              env, total_tiles, tiling, tw, cfg)

# ── Train ────────────────────────────────────────────────────────────────

print('\nTraining...')
try:
    for i in (pbar := tqdm(range(cfg.n_episodes), desc=cfg.label)):
        _run_single_episode(env, ag, actor, critic, [pc], cfg)
        sf = np.mean(np.array(env.episodes['meta_info'][-100:]) == 'completed')
        et = np.mean(env.episodes['duration'][-100:])
        pbar.set_description(f'{cfg.label} | success: {sf:.2f}, time: {et:.1f}')
        if sf > 0.99 and i > 10:
            break
except KeyboardInterrupt:
    print('Interrupted')

# ── After learning ───────────────────────────────────────────────────────

print('\nRecording AFTER learning...')
acts_after = record_activations(critic_nn, pypi_fta, env, ag, pc)
print(f'Recorded {len(acts_after)} locations')

pt_after = compute_pypi_per_tile(acts_after, total_tiles, pre_fta_dim)
save_heatmaps(pt_after, os.path.join(OUT_DIR, 'per_tile_sparsity_pypi_ln_norelu_after.pdf'),
              env, total_tiles, tiling, tw, cfg)

print('\nDone!')
