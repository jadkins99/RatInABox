"""PyPI FTA: Per-feature heatmaps (one per input dimension, summing over tiles).

Architecture: Linear(50->20) -> LayerNorm(20) -> FTA -> Linear(220->1)
FTA output shape: [batch, num_tiles * features] = [1, 11 * 20]
Reshaped: [1, 11, 20] — we sum across axis=1 (tiles) to get [1, 20].
Each of the 20 heatmaps shows total FTA activation for that feature across space.
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
PRE_FTA_DIM = 20

pypi_fta = PyPiFTA(
    bound=BOUND,
    spillover_base=0,
    spillover_mode='derive_from_tile_width',
    tile_width=None,
    num_tiles=N_TILES,
)
total_tiles = pypi_fta.num_tiles  # 11
fta_out_dim = PRE_FTA_DIM * total_tiles  # 220

print(f'PyPI FTA: num_tiles={total_tiles}, tile_width={pypi_fta.tile_width:.3f}, '
      f'spillover={pypi_fta.spillover}')

class BackboneWrapper(nn.Module):
    def __init__(self, seq):
        super().__init__()
        self.net = seq
    def forward(self, x):
        return self.net(x)

# Linear -> LayerNorm -> FTA (no ReLU)
critic_nn = BackboneWrapper(nn.Sequential(
    nn.Linear(50, PRE_FTA_DIM),
    nn.LayerNorm(PRE_FTA_DIM),
    pypi_fta,
    nn.Linear(fta_out_dim, 1),
))
print(f'\nCritic:\n{critic_nn}')

actor_nn = VxVyGaussianHead(Backbone(n_in=50, n_out=2, hidden=[50]))

cfg = ExperimentConfig(label='PyPI FTA Linear+LN', n_episodes=500)
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


def compute_per_feature(fta_acts, total_tiles, n_features):
    """One sparsity map per input feature, summing FTA activation across tiles.

    FTA output reshaped: [batch, total_tiles, n_features]
    For each feature f, sum across tiles -> scalar per observation.
    """
    per_feature = {}
    for feat_idx in range(n_features):
        sp = {}
        for obs_key, tensors in fta_acts.items():
            vals = []
            for t in tensors:
                # [1, total_tiles * n_features] -> [total_tiles, n_features]
                arr = t.numpy().reshape(-1, total_tiles, n_features)
                # sum over tiles for this feature
                feat_sum = arr[0, :, feat_idx].sum()
                vals.append(float(feat_sum))
            sp[obs_key] = sum(vals) / len(vals)
        per_feature[feat_idx] = sp
    return per_feature


def save_per_feature_pdf(per_feature, n_features, env, cfg, pdf_path):
    with PdfPages(pdf_path) as pdf:
        for feat_idx in range(n_features):
            sp = per_feature[feat_idx]
            title = f'Feature {feat_idx+1}/{n_features} (sum over {total_tiles} tiles)'
            fig, ax = plot_sparsity_map(env, sp, bins=40, title=title)
            display_reward_patch(fig, ax, cfg.goal_pos, cfg.goal_radius)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f'Saved {n_features} heatmaps -> {pdf_path}')


# ── Before learning ─────────────────────────────────────────────────────

print('\nRecording BEFORE learning...')
acts_before = record_activations(critic_nn, pypi_fta, env, ag, pc)
print(f'Recorded {len(acts_before)} locations')

pf_before = compute_per_feature(acts_before, total_tiles, PRE_FTA_DIM)
save_per_feature_pdf(pf_before, PRE_FTA_DIM, env, cfg,
                     os.path.join(OUT_DIR, 'per_feature_sparsity_before.pdf'))

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

pf_after = compute_per_feature(acts_after, total_tiles, PRE_FTA_DIM)
save_per_feature_pdf(pf_after, PRE_FTA_DIM, env, cfg,
                     os.path.join(OUT_DIR, 'per_feature_sparsity_after.pdf'))

print('\nDone!')
