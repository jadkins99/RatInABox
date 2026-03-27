"""Redo post-FTA ReLU heatmaps collapsed to 20 features (sum over 11 tiles per feature).

Architecture: Linear(50->20) -> LayerNorm(20) -> FTA -> ReLU -> Linear(220->1)
ReLU output: [1, 220] reshaped to [1, 11, 20], sum over tiles -> [1, 20]
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
TILE_WIDTH = 2 * BOUND / N_TILES
PRE_FTA_DIM = 20

pypi_fta = PyPiFTA(
    bound=BOUND, spillover_base=0, spillover_mode='derive_from_tile_width',
    tile_width=None, num_tiles=N_TILES,
)
total_tiles = pypi_fta.num_tiles  # 11
fta_out_dim = PRE_FTA_DIM * total_tiles  # 220

class BackboneWrapper(nn.Module):
    def __init__(self, seq):
        super().__init__()
        self.net = seq
    def forward(self, x):
        return self.net(x)

post_fta_relu = nn.ReLU()

critic_nn = BackboneWrapper(nn.Sequential(
    nn.Linear(50, PRE_FTA_DIM),
    nn.LayerNorm(PRE_FTA_DIM),
    pypi_fta,
    post_fta_relu,
    nn.Linear(fta_out_dim, 1),
))
print(f'{critic_nn}')

actor_nn = VxVyGaussianHead(Backbone(n_in=50, n_out=2, hidden=[50]))

cfg = ExperimentConfig(label='FTA+ReLU', n_episodes=500)
env, ag = _make_env_and_agent(cfg)
pc = PlaceCells(ag, params={'n': 50})

opt_fn = lambda p: torch.optim.SGD(p, lr=cfg.eta, maximize=True)
actor = Actor(ag, params={'input_layers': [pc], 'NeuralNetworkModule': actor_nn,
                          'tau': cfg.tau, 'tau_z': cfg.tau_e, 'optimizer': opt_fn})
critic = Critic(ag, params={'input_layers': [pc], 'NeuralNetworkModule': critic_nn,
                            'tau': cfg.tau, 'tau_z': cfg.tau_e, 'optimizer': opt_fn})

# ── Helpers ──────────────────────────────────────────────────────────────

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


def compute_per_feature_collapsed(acts, total_tiles, n_features):
    """Collapse [1, total_tiles * n_features] -> sum over tiles -> 20 features."""
    per_feature = {}
    for feat_idx in range(n_features):
        sp = {}
        for obs_key, tensors in acts.items():
            vals = []
            for t in tensors:
                arr = t.numpy().reshape(-1, total_tiles, n_features)
                feat_sum = arr[0, :, feat_idx].sum()
                vals.append(float(feat_sum))
            sp[obs_key] = sum(vals) / len(vals)
        per_feature[feat_idx] = sp
    return per_feature


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


# ── Before learning ─────────────────────────────────────────────────────

print('\nRecording BEFORE learning...')
acts_before = record_activations(critic_nn, {'relu': post_fta_relu}, env, ag, pc)
print(f'Recorded {len(acts_before["relu"])} locations')

pf_relu_before = compute_per_feature_collapsed(acts_before['relu'], total_tiles, PRE_FTA_DIM)
save_heatmaps_pdf(pf_relu_before, PRE_FTA_DIM, env, cfg,
    os.path.join(OUT_DIR, 'post_fta_relu_collapsed_before.pdf'),
    lambda i, n: f'Feature {i+1}/{n} (post-FTA ReLU)')

# ── Train ────────────────────────────────────────────────────────────────

print('\nTraining...')
try:
    for i in (pbar := tqdm(range(cfg.n_episodes), desc=cfg.label)):
        _run_single_episode(env, ag, actor, critic, [pc], cfg)
        sf = np.mean(np.array(env.episodes['meta_info'][-100:]) == 'completed')
        et = np.mean(env.episodes['duration'][-100:])
        pbar.set_description(f'{cfg.label} | success: {sf:.2f}, time: {et:.1f}')
        if sf > 0.99 and i > 10: break
except KeyboardInterrupt:
    print('Interrupted')

# ── After learning ───────────────────────────────────────────────────────

print('\nRecording AFTER learning...')
acts_after = record_activations(critic_nn, {'relu': post_fta_relu}, env, ag, pc)
print(f'Recorded {len(acts_after["relu"])} locations')

pf_relu_after = compute_per_feature_collapsed(acts_after['relu'], total_tiles, PRE_FTA_DIM)
save_heatmaps_pdf(pf_relu_after, PRE_FTA_DIM, env, cfg,
    os.path.join(OUT_DIR, 'post_fta_relu_collapsed_after.pdf'),
    lambda i, n: f'Feature {i+1}/{n} (post-FTA ReLU)')

print('\nDone!')
