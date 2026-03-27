"""Run per-tile sparsity experiments with LayerNorm + eta=delta.

Generates 4 PDFs:
  - per_tile_sparsity_ln_before.pdf   (our FTA, before learning)
  - per_tile_sparsity_ln_after.pdf    (our FTA, after learning)
  - per_tile_sparsity_pypi_ln_before.pdf  (PyPI FTA, before learning)
  - per_tile_sparsity_pypi_ln_after.pdf   (PyPI FTA, after learning)
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
from experiments import ExperimentConfig, run_experiment, _make_env_and_agent, _run_single_episode
from activation_recorder import ActivationRecorder, find_fta_module
from viz import plot_sparsity_map, display_reward_patch
from networks import Backbone, VxVyGaussianHead, make_fta_critic

OUT_DIR = os.path.join(os.path.dirname(__file__), 'mnt', 'RatInABox')

# ── Helpers ──────────────────────────────────────────────────────────────

def record_activations(critic_net, fta_module, env, ag, placecells, timesteps=2000):
    """Record FTA activations over `timesteps` of inference."""
    critic_net.eval()
    recorder = ActivationRecorder()
    recorder.attach(fta_module, name='fta')

    start_t = env.t
    while env.t < timesteps + start_t:
        obs, rr, term, _, info = env.step1()
        if env.t - env.episodes['start'][-1] > 15:
            env.reset(episode_meta_info='timeout')
        elif term:
            env.reset(episode_meta_info='completed')
        recorder.set_observation(obs)
        state = torch.tensor(placecells.get_state(), dtype=torch.float).t()
        with torch.inference_mode():
            _ = critic_net(state)
        for c in [placecells]:
            c.update()

    acts = recorder.get('fta')
    recorder.detach_all()
    critic_net.train()
    return acts


def compute_per_tile_sparsity(fta_acts, input_dim, n_tiles, n_tilings, threshold=0.01):
    results = {}
    for tiling_idx in range(n_tilings):
        for tile_idx in range(n_tiles):
            sp = {}
            for obs_key, tensors in fta_acts.items():
                vals = []
                for t in tensors:
                    arr = t.numpy()
                    if n_tilings == 1:
                        reshaped = arr.reshape(-1, input_dim, n_tiles)
                        tile_act = reshaped[0, :, tile_idx]
                    else:
                        reshaped = arr.reshape(-1, input_dim, n_tiles, n_tilings)
                        tile_act = reshaped[0, :, tile_idx, tiling_idx]
                    above = tile_act[tile_act > threshold]
                    vals.append(float(np.sum(above) / tile_act.size) if tile_act.size > 0 else 0.0)
                sp[obs_key] = sum(vals) / len(vals)
            results[(tiling_idx, tile_idx)] = sp
    return results


def save_heatmaps_pdf(per_tile, fta_module, env, cfg, pdf_path, n_tiles, n_tilings):
    tile_centers = fta_module.c_vec.cpu().numpy()
    tile_delta = fta_module.tile_delta.item()

    with PdfPages(pdf_path) as pdf:
        for tiling_idx in range(n_tilings):
            for tile_idx in range(n_tiles):
                sp = per_tile[(tiling_idx, tile_idx)]
                center = tile_centers[tile_idx]
                lo, hi = center, center + tile_delta
                title = (f'Tiling {tiling_idx+1}/{n_tilings}, '
                         f'Tile {tile_idx+1}/{n_tiles} [{lo:.2f}, {hi:.2f})')
                fig, ax = plot_sparsity_map(env, sp, bins=40, title=title)
                display_reward_patch(fig, ax, cfg.goal_pos, cfg.goal_radius)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    print(f'Saved {n_tiles * n_tilings} heatmaps → {pdf_path}')


# ── Our FTA experiments ──────────────────────────────────────────────────

print("=" * 60)
print("OUR FTA: LayerNorm + eta=delta (n_tiles=10)")
print("=" * 60)

N_TILES = 10
N_TILINGS = 1

cfg = ExperimentConfig(
    label=f'FTA LN+eta=delta tiles={N_TILES}',
    critic_type='fta',
    # fta_eta defaults to None → eta=delta in make_fta_critic
    fta_input_min=0.0,
    fta_input_max=1.0,
    fta_n_tiles=N_TILES,
    fta_n_tilings=N_TILINGS,
    fta_post_fta=[1],
    n_episodes=500,
)

# Build env + networks for "before learning" recording
result = run_experiment(cfg, ActorCls=Actor, CriticCls=Critic)

# We need before-learning activations, so build a fresh critic to record from
# Actually: record from result but BEFORE training we need a fresh one.
# Simpler: build fresh env+agent, record, then train.

# --- Fresh setup for before-learning ---
from experiments import _make_env_and_agent, _make_networks

env_b, ag_b = _make_env_and_agent(cfg)
pc_b = PlaceCells(ag_b, params={'n': cfg.n_place_cells})
critic_nn_b, actor_nn_b = _make_networks(cfg, n_in=pc_b.n)

# Verify LayerNorm + eta=delta
print(f"\nCritic architecture:\n{critic_nn_b}")
fta_b = find_fta_module(critic_nn_b)
print(f"eta={fta_b.fta_eta}, delta={fta_b.tile_delta.item():.4f}, eta==delta: {abs(fta_b.fta_eta - fta_b.tile_delta.item()) < 1e-6}")

opt_fn = lambda p: torch.optim.SGD(p, lr=cfg.eta, maximize=True)
actor_b = Actor(ag_b, params={'input_layers': [pc_b], 'NeuralNetworkModule': actor_nn_b,
                               'tau': cfg.tau, 'tau_z': cfg.tau_e, 'optimizer': opt_fn})
critic_b = Critic(ag_b, params={'input_layers': [pc_b], 'NeuralNetworkModule': critic_nn_b,
                                 'tau': cfg.tau, 'tau_z': cfg.tau_e, 'optimizer': opt_fn})

# Before learning
print("\nRecording BEFORE learning...")
acts_before = record_activations(critic_nn_b, fta_b, env_b, ag_b, pc_b, timesteps=2000)
print(f"Recorded {len(acts_before)} locations")

per_tile_before = compute_per_tile_sparsity(acts_before, fta_b.input_dim, N_TILES, N_TILINGS)
save_heatmaps_pdf(per_tile_before, fta_b, env_b, cfg,
                  os.path.join(OUT_DIR, 'per_tile_sparsity_ln_before.pdf'), N_TILES, N_TILINGS)

# Train
print("\nTraining...")
try:
    for i in (pbar := tqdm(range(cfg.n_episodes), desc=cfg.label)):
        _run_single_episode(env_b, ag_b, actor_b, critic_b, [pc_b], cfg)
        sf = np.mean(np.array(env_b.episodes['meta_info'][-100:]) == 'completed')
        et = np.mean(env_b.episodes['duration'][-100:])
        pbar.set_description(f'{cfg.label} | success: {sf:.2f}, time: {et:.1f}')
        if sf > 0.99 and i > 10:
            break
except KeyboardInterrupt:
    print('Interrupted')

# After learning
print("\nRecording AFTER learning...")
acts_after = record_activations(critic_nn_b, fta_b, env_b, ag_b, pc_b, timesteps=2000)
print(f"Recorded {len(acts_after)} locations")

per_tile_after = compute_per_tile_sparsity(acts_after, fta_b.input_dim, N_TILES, N_TILINGS)
save_heatmaps_pdf(per_tile_after, fta_b, env_b, cfg,
                  os.path.join(OUT_DIR, 'per_tile_sparsity_ln_after.pdf'), N_TILES, N_TILINGS)


# ── PyPI FTA experiments ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PyPI FTA: LayerNorm + eta=delta (n_tiles=10)")
print("=" * 60)

# Load PyPI FTA via importlib
spec = importlib.util.spec_from_file_location(
    'fta_pypi_torch',
    '/sessions/focused-laughing-newton/.local/lib/python3.10/site-packages/fta/torch.py'
)
fta_pypi_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fta_pypi_mod)
PyPiFTA = fta_pypi_mod.FTA

BOUND = 1.0
# tile_width = 2*bound/num_tiles = 2*1.0/10 = 0.2
# spillover = tile_width = 0.2
TILE_WIDTH = 2 * BOUND / N_TILES  # 0.2
pypi_fta = PyPiFTA(
    bound=BOUND,
    spillover_base=0,   # eta = delta = tile_width
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

def make_pypi_critic():
    return BackboneWrapper(nn.Sequential(
        nn.Linear(50, pre_fta_dim),
        nn.ReLU(),
        nn.LayerNorm(pre_fta_dim),   # LayerNorm before FTA
        pypi_fta,
        nn.Linear(fta_out_dim, 1),
    ))

cfg_pypi = ExperimentConfig(
    label=f'PyPI FTA LN+eta=delta tiles={N_TILES}',
    n_episodes=500,
)

env_p, ag_p = _make_env_and_agent(cfg_pypi)
pc_p = PlaceCells(ag_p, params={'n': 50})

critic_nn_p = make_pypi_critic()
actor_nn_p = VxVyGaussianHead(Backbone(n_in=50, n_out=2, hidden=[50]))

opt_fn_p = lambda p: torch.optim.SGD(p, lr=cfg_pypi.eta, maximize=True)
actor_p = Actor(ag_p, params={'input_layers': [pc_p], 'NeuralNetworkModule': actor_nn_p,
                               'tau': cfg_pypi.tau, 'tau_z': cfg_pypi.tau_e, 'optimizer': opt_fn_p})
critic_p = Critic(ag_p, params={'input_layers': [pc_p], 'NeuralNetworkModule': critic_nn_p,
                                 'tau': cfg_pypi.tau, 'tau_z': cfg_pypi.tau_e, 'optimizer': opt_fn_p})

# Before learning
print("\nRecording BEFORE learning (PyPI)...")
critic_nn_p.eval()
rec_p = ActivationRecorder()
rec_p.attach(pypi_fta, name='fta')

start_t = env_p.t
while env_p.t < 2000 + start_t:
    obs, rr, term, _, info = env_p.step1()
    if env_p.t - env_p.episodes['start'][-1] > 15:
        env_p.reset(episode_meta_info='timeout')
    elif term:
        env_p.reset(episode_meta_info='completed')
    rec_p.set_observation(obs)
    state = torch.tensor(pc_p.get_state(), dtype=torch.float).t()
    with torch.inference_mode():
        _ = critic_nn_p(state)
    for c in [pc_p]: c.update()

fta_acts_pypi_before = rec_p.get('fta')
rec_p.detach_all()
critic_nn_p.train()
print(f"Recorded {len(fta_acts_pypi_before)} locations")

# Save PyPI before-learning heatmaps
total_tiles_pypi = pypi_fta.num_tiles
tiling_pypi = pypi_fta._tiling.cpu().numpy()
tw = pypi_fta.tile_width

def save_pypi_heatmaps(per_tile_dict, pdf_path, env, total_tiles, tiling, tw, cfg):
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
    print(f'Saved {total_tiles} heatmaps → {pdf_path}')


def compute_pypi_per_tile(fta_acts, total_tiles, pre_fta_dim):
    per_tile = {}
    for tile_idx in range(total_tiles):
        sp = {}
        for obs_key, tensors in fta_acts.items():
            vals = []
            for t in tensors:
                arr = t.numpy().reshape(-1, total_tiles, pre_fta_dim)
                tile_act = arr[0, tile_idx, :]
                above = tile_act[tile_act > 0.01]
                vals.append(float(np.sum(above) / tile_act.size) if tile_act.size > 0 else 0.0)
            sp[obs_key] = sum(vals) / len(vals)
        per_tile[tile_idx] = sp
    return per_tile

pt_pypi_before = compute_pypi_per_tile(fta_acts_pypi_before, total_tiles_pypi, pre_fta_dim)
save_pypi_heatmaps(pt_pypi_before, os.path.join(OUT_DIR, 'per_tile_sparsity_pypi_ln_before.pdf'),
                   env_p, total_tiles_pypi, tiling_pypi, tw, cfg_pypi)

# Train PyPI
print("\nTraining (PyPI)...")
try:
    for i in (pbar := tqdm(range(cfg_pypi.n_episodes), desc=cfg_pypi.label)):
        _run_single_episode(env_p, ag_p, actor_p, critic_p, [pc_p], cfg_pypi)
        sf = np.mean(np.array(env_p.episodes['meta_info'][-100:]) == 'completed')
        et = np.mean(env_p.episodes['duration'][-100:])
        pbar.set_description(f'{cfg_pypi.label} | success: {sf:.2f}, time: {et:.1f}')
        if sf > 0.99 and i > 10:
            break
except KeyboardInterrupt:
    print('Interrupted')

# After learning
print("\nRecording AFTER learning (PyPI)...")
critic_nn_p.eval()
rec_p2 = ActivationRecorder()
rec_p2.attach(pypi_fta, name='fta')

start_t = env_p.t
while env_p.t < 2000 + start_t:
    obs, rr, term, _, info = env_p.step1()
    if env_p.t - env_p.episodes['start'][-1] > 15:
        env_p.reset(episode_meta_info='timeout')
    elif term:
        env_p.reset(episode_meta_info='completed')
    rec_p2.set_observation(obs)
    state = torch.tensor(pc_p.get_state(), dtype=torch.float).t()
    with torch.inference_mode():
        _ = critic_nn_p(state)
    for c in [pc_p]: c.update()

fta_acts_pypi_after = rec_p2.get('fta')
rec_p2.detach_all()
print(f"Recorded {len(fta_acts_pypi_after)} locations")

pt_pypi_after = compute_pypi_per_tile(fta_acts_pypi_after, total_tiles_pypi, pre_fta_dim)
save_pypi_heatmaps(pt_pypi_after, os.path.join(OUT_DIR, 'per_tile_sparsity_pypi_ln_after.pdf'),
                   env_p, total_tiles_pypi, tiling_pypi, tw, cfg_pypi)

print("\n" + "=" * 60)
print("All 4 PDFs generated!")
print("=" * 60)
