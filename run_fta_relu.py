"""Three sets of heatmaps (before & after learning):
1. FTA+ReLU agent: 20 per-feature heatmaps (sum over tiles from FTA output)
2. FTA+ReLU agent: heatmap of the ReLU layer that follows FTA (220 -> 220)
3. Baseline agent (no FTA): heatmap of a ReLU layer

Architecture (FTA+ReLU):
  Linear(50->20) -> LayerNorm(20) -> FTA -> ReLU -> Linear(220->1)

Architecture (Baseline):
  Linear(50->20) -> ReLU -> Linear(20->1)
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
TILE_WIDTH = 2 * BOUND / N_TILES  # 0.2
PRE_FTA_DIM = 20

# ── Helpers ──────────────────────────────────────────────────────────────

class BackboneWrapper(nn.Module):
    def __init__(self, seq):
        super().__init__()
        self.net = seq
    def forward(self, x):
        return self.net(x)


def record_activations(net, modules_dict, env, ag, placecells, timesteps=2000):
    """Record activations from multiple named modules simultaneously."""
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
    """One map per input feature, summing FTA activation across tiles."""
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
# FTA + ReLU agent
# ══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("FTA + ReLU agent")
print("=" * 60)

pypi_fta = PyPiFTA(
    bound=BOUND, spillover_base=0, spillover_mode='derive_from_tile_width',
    tile_width=None, num_tiles=N_TILES,
)
total_tiles = pypi_fta.num_tiles  # 11
fta_out_dim = PRE_FTA_DIM * total_tiles  # 220

post_fta_relu = nn.ReLU()

critic_fta = BackboneWrapper(nn.Sequential(
    nn.Linear(50, PRE_FTA_DIM),
    nn.LayerNorm(PRE_FTA_DIM),
    pypi_fta,
    post_fta_relu,
    nn.Linear(fta_out_dim, 1),
))
print(f'\n{critic_fta}')

actor_fta = VxVyGaussianHead(Backbone(n_in=50, n_out=2, hidden=[50]))

cfg_fta = ExperimentConfig(label='FTA+ReLU', n_episodes=500)
env_f, ag_f = _make_env_and_agent(cfg_fta)
pc_f = PlaceCells(ag_f, params={'n': 50})

opt_fn = lambda p: torch.optim.SGD(p, lr=cfg_fta.eta, maximize=True)
actor_f = Actor(ag_f, params={'input_layers': [pc_f], 'NeuralNetworkModule': actor_fta,
                              'tau': cfg_fta.tau, 'tau_z': cfg_fta.tau_e, 'optimizer': opt_fn})
critic_f = Critic(ag_f, params={'input_layers': [pc_f], 'NeuralNetworkModule': critic_fta,
                                'tau': cfg_fta.tau, 'tau_z': cfg_fta.tau_e, 'optimizer': opt_fn})

# Before learning
print('\nRecording BEFORE learning (FTA+ReLU)...')
acts_before = record_activations(critic_fta, {'fta': pypi_fta, 'relu': post_fta_relu},
                                  env_f, ag_f, pc_f)
print(f'Recorded {len(acts_before["fta"])} locations')

# 1. Per-feature FTA heatmaps
pf_before = compute_per_feature(acts_before['fta'], total_tiles, PRE_FTA_DIM)
save_heatmaps_pdf(pf_before, PRE_FTA_DIM, env_f, cfg_fta,
    os.path.join(OUT_DIR, 'per_feature_fta_before.pdf'),
    lambda i, n: f'Feature {i+1}/{n}')

# 2. Post-FTA ReLU heatmaps (220 units — save all 220)
relu_before = compute_per_unit(acts_before['relu'], fta_out_dim)
save_heatmaps_pdf(relu_before, fta_out_dim, env_f, cfg_fta,
    os.path.join(OUT_DIR, 'post_fta_relu_before.pdf'),
    lambda i, n: f'ReLU unit {i+1}/{n}')

# Train
print('\nTraining (FTA+ReLU)...')
try:
    for i in (pbar := tqdm(range(cfg_fta.n_episodes), desc=cfg_fta.label)):
        _run_single_episode(env_f, ag_f, actor_f, critic_f, [pc_f], cfg_fta)
        sf = np.mean(np.array(env_f.episodes['meta_info'][-100:]) == 'completed')
        et = np.mean(env_f.episodes['duration'][-100:])
        pbar.set_description(f'{cfg_fta.label} | success: {sf:.2f}, time: {et:.1f}')
        if sf > 0.99 and i > 10: break
except KeyboardInterrupt:
    print('Interrupted')

# After learning
print('\nRecording AFTER learning (FTA+ReLU)...')
acts_after = record_activations(critic_fta, {'fta': pypi_fta, 'relu': post_fta_relu},
                                 env_f, ag_f, pc_f)

pf_after = compute_per_feature(acts_after['fta'], total_tiles, PRE_FTA_DIM)
save_heatmaps_pdf(pf_after, PRE_FTA_DIM, env_f, cfg_fta,
    os.path.join(OUT_DIR, 'per_feature_fta_after.pdf'),
    lambda i, n: f'Feature {i+1}/{n}')

relu_after = compute_per_unit(acts_after['relu'], fta_out_dim)
save_heatmaps_pdf(relu_after, fta_out_dim, env_f, cfg_fta,
    os.path.join(OUT_DIR, 'post_fta_relu_after.pdf'),
    lambda i, n: f'ReLU unit {i+1}/{n}')


# ══════════════════════════════════════════════════════════════════════════
# Baseline agent (no FTA)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Baseline agent (no FTA)")
print("=" * 60)

baseline_relu = nn.ReLU()

critic_base = BackboneWrapper(nn.Sequential(
    nn.Linear(50, PRE_FTA_DIM),
    baseline_relu,
    nn.Linear(PRE_FTA_DIM, 1),
))
print(f'\n{critic_base}')

actor_base = VxVyGaussianHead(Backbone(n_in=50, n_out=2, hidden=[50]))

cfg_base = ExperimentConfig(label='Baseline (no FTA)', n_episodes=500)
env_b, ag_b = _make_env_and_agent(cfg_base)
pc_b = PlaceCells(ag_b, params={'n': 50})

opt_fn_b = lambda p: torch.optim.SGD(p, lr=cfg_base.eta, maximize=True)
actor_b = Actor(ag_b, params={'input_layers': [pc_b], 'NeuralNetworkModule': actor_base,
                              'tau': cfg_base.tau, 'tau_z': cfg_base.tau_e, 'optimizer': opt_fn_b})
critic_b = Critic(ag_b, params={'input_layers': [pc_b], 'NeuralNetworkModule': critic_base,
                                'tau': cfg_base.tau, 'tau_z': cfg_base.tau_e, 'optimizer': opt_fn_b})

# Before learning
print('\nRecording BEFORE learning (Baseline)...')
base_acts_before = record_activations(critic_base, {'relu': baseline_relu},
                                       env_b, ag_b, pc_b)
print(f'Recorded {len(base_acts_before["relu"])} locations')

base_relu_before = compute_per_unit(base_acts_before['relu'], PRE_FTA_DIM)
save_heatmaps_pdf(base_relu_before, PRE_FTA_DIM, env_b, cfg_base,
    os.path.join(OUT_DIR, 'baseline_relu_before.pdf'),
    lambda i, n: f'ReLU unit {i+1}/{n}')

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

# After learning
print('\nRecording AFTER learning (Baseline)...')
base_acts_after = record_activations(critic_base, {'relu': baseline_relu},
                                      env_b, ag_b, pc_b)

base_relu_after = compute_per_unit(base_acts_after['relu'], PRE_FTA_DIM)
save_heatmaps_pdf(base_relu_after, PRE_FTA_DIM, env_b, cfg_base,
    os.path.join(OUT_DIR, 'baseline_relu_after.pdf'),
    lambda i, n: f'ReLU unit {i+1}/{n}')

print('\nDone! Generated 6 PDFs.')
