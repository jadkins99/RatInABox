"""Corrected baseline: Linear(50->220) -> ReLU -> Linear(220->20) -> ReLU -> Linear(20->1)

Records the second ReLU (20 units) for comparison with FTA agent's post-compression ReLU.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mnt', 'RatInABox'))

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

PRE_FTA_DIM = 20

class BackboneWrapper(nn.Module):
    def __init__(self, seq):
        super().__init__()
        self.net = seq
    def forward(self, x):
        return self.net(x)


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


def compute_per_unit(acts, n_units):
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
# Baseline: Linear(50->220) -> ReLU -> Linear(220->20) -> ReLU -> Linear(20->1)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Baseline: Linear(50->220) -> ReLU -> Linear(220->20) -> ReLU -> Linear(20->1)")
print("=" * 60)

baseline_relu2 = nn.ReLU()  # the second ReLU we record (20 units)

critic_base = BackboneWrapper(nn.Sequential(
    nn.Linear(50, 220),       # 0: widen to match FTA output dimension
    nn.ReLU(),                 # 1: first ReLU
    nn.Linear(220, PRE_FTA_DIM),  # 2: compress 220 -> 20
    baseline_relu2,            # 3: second ReLU (recorded)
    nn.Linear(PRE_FTA_DIM, 1), # 4: readout
))
print(f'\n{critic_base}')

actor_base_nn = VxVyGaussianHead(Backbone(n_in=50, n_out=2, hidden=[50]))

cfg_base = ExperimentConfig(label='Baseline (no FTA)', n_episodes=500)
env_b, ag_b = _make_env_and_agent(cfg_base)
pc_b = PlaceCells(ag_b, params={'n': 50})

opt_fn = lambda p: torch.optim.SGD(p, lr=cfg_base.eta, maximize=True)
actor_b = Actor(ag_b, params={'input_layers': [pc_b], 'NeuralNetworkModule': actor_base_nn,
                              'tau': cfg_base.tau, 'tau_z': cfg_base.tau_e, 'optimizer': opt_fn})
critic_b = Critic(ag_b, params={'input_layers': [pc_b], 'NeuralNetworkModule': critic_base,
                                'tau': cfg_base.tau, 'tau_z': cfg_base.tau_e, 'optimizer': opt_fn})

# Before learning
print('\nRecording BEFORE learning...')
acts_before = record_activations(critic_base, {'relu': baseline_relu2}, env_b, ag_b, pc_b)
print(f'Recorded {len(acts_before["relu"])} locations')

relu_before = compute_per_unit(acts_before['relu'], PRE_FTA_DIM)
save_heatmaps_pdf(relu_before, PRE_FTA_DIM, env_b, cfg_base,
    os.path.join(OUT_DIR, 'baseline_relu_before.pdf'),
    lambda i, n: f'ReLU unit {i+1}/{n}')

# Train
print('\nTraining...')
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
print('\nRecording AFTER learning...')
acts_after = record_activations(critic_base, {'relu': baseline_relu2}, env_b, ag_b, pc_b)
print(f'Recorded {len(acts_after["relu"])} locations')

relu_after = compute_per_unit(acts_after['relu'], PRE_FTA_DIM)
save_heatmaps_pdf(relu_after, PRE_FTA_DIM, env_b, cfg_base,
    os.path.join(OUT_DIR, 'baseline_relu_after.pdf'),
    lambda i, n: f'ReLU unit {i+1}/{n}')

print('\nDone!')
