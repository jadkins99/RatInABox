"""Decoding position from raw cells vs learned FTA/baseline representations.

Usage: python run_decoding.py <cell_type> [--seeds N]
  cell_type: 'place', 'grid', or 'bvc'
  --seeds N: number of seeds per N value (default 5)

Compares linear decodeability (Ridge regression) of:
  1. Raw cell firing rates (N-dim)
  2. FTA 220-dim post-FTA representation
  3. Baseline 20-dim ReLU1 representation

FTA architecture:
  Linear(N→20) → LayerNorm(non-adaptive) → FTA → Linear(220→20) → ReLU → Linear(20→1)
Baseline architecture:
  Linear(N→20) → ReLU1 → Linear(20→20) → ReLU2 → Linear(20→1)

Outputs:
  decoding_<cell_type>_results.npz  — decoding errors + success rates
  decoding_<cell_type>_linear.pdf   — linear decodeability plot
"""

import argparse, os, importlib
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import Ridge

import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells, GridCells, BoundaryVectorCells
from ratinabox.contribs.TaskEnvironment import (
    SpatialGoalEnvironment, SpatialGoal, Reward,
)

from ac import Actor, Critic
from experiments import ExperimentConfig, _run_single_episode
from networks import Backbone, VxVyGaussianHead

# ── Load PyPI FTA ────────────────────────────────────────────────────────
import importlib.metadata as _meta
_fta_dist = _meta.distribution('fuzzy-tiling-activation')
_fta_pkg_dir = str(_fta_dist._path.parent)
_fta_torch_path = os.path.join(_fta_pkg_dir, 'fta', 'torch.py')
spec = importlib.util.spec_from_file_location('fta_pypi_torch', _fta_torch_path)
fta_pypi_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fta_pypi_mod)
PyPiFTA = fta_pypi_mod.FTA

# ── CLI ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('cell_type', nargs='?', default='place',
                    choices=['place', 'grid', 'bvc'])
parser.add_argument('--seeds', type=int, default=5)
args = parser.parse_args()

CELL_TYPE = args.cell_type
N_REPEATS = args.seeds

# ── Constants ────────────────────────────────────────────────────────────
N_FEATURES = [80, 40, 20, 10]
PRE_FTA_DIM = 20
N_TILES = 10
BOUND = 1.0
DT = 0.1
TRAIN_DURATION = 5 * 60   # seconds
TEST_DURATION = 1 * 60    # seconds
N_EPISODES = 2000
GOAL_POS = np.array([0.5, 0.5])
GOAL_RADIUS = 0.1
WALL = np.array([[0.4, 0], [0.4, 0.4]])
TAU = 5.0
TAU_E = 5.0
ETA = 0.002

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Helpers ──────────────────────────────────────────────────────────────

def make_cells(agent, cell_type, n):
    if cell_type == 'place':
        return PlaceCells(agent, params={
            'n': n, 'description': 'gaussian_threshold', 'widths': 0.4,
        })
    elif cell_type == 'grid':
        return GridCells(agent, params={'n': n})
    elif cell_type == 'bvc':
        return BoundaryVectorCells(agent, params={'n': n})
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")


def make_task_env():
    env = SpatialGoalEnvironment(
        dt=DT, teleport_on_reset=True, episode_terminate_delay=1.0,
    )
    env.add_wall(WALL)
    reward = Reward(1.0, decay='none', expire_clock=1.0, dt=DT)
    goals = [SpatialGoal(env, pos=GOAL_POS, goal_radius=GOAL_RADIUS, reward=reward)]
    env.goal_cache.reset_goals = goals
    ag = Agent(env, params={'dt': DT})
    env.add_agents(ag)
    return env, ag


def make_fta_critic(n_in):
    """Linear(n_in→20) → LN (non-adaptive) → FTA → Linear(220→20) → ReLU → Linear(20→1)"""
    fta = PyPiFTA(
        bound=BOUND, spillover_base=0,
        spillover_mode='derive_from_tile_width',
        tile_width=None, num_tiles=N_TILES,
    )
    total_tiles = fta.num_tiles  # 11
    fta_out_dim = PRE_FTA_DIM * total_tiles  # 220
    net = nn.Sequential(
        nn.Linear(n_in, PRE_FTA_DIM),
        nn.LayerNorm(PRE_FTA_DIM, elementwise_affine=False),
        fta,
        nn.Linear(fta_out_dim, PRE_FTA_DIM),
        nn.ReLU(),
        nn.Linear(PRE_FTA_DIM, 1),
    )
    return net, fta


def make_baseline_critic(n_in):
    """Linear(n_in→20) → ReLU1 → Linear(20→20) → ReLU2 → Linear(20→1)"""
    relu1 = nn.ReLU()
    net = nn.Sequential(
        nn.Linear(n_in, PRE_FTA_DIM),
        relu1,
        nn.Linear(PRE_FTA_DIM, PRE_FTA_DIM),
        nn.ReLU(),
        nn.Linear(PRE_FTA_DIM, 1),
    )
    return net, relu1


def train_agent(env, ag, cells, critic_net, n_in):
    """Train an actor-critic agent for N_EPISODES, return (critic, actor, success_rate)."""
    actor_nn = VxVyGaussianHead(Backbone(n_in=n_in, n_out=2, hidden=[50]))
    opt_fn = lambda p: torch.optim.SGD(p, lr=ETA, maximize=True)

    actor = Actor(ag, params={
        'input_layers': [cells],
        'NeuralNetworkModule': actor_nn,
        'tau': TAU, 'tau_z': TAU_E, 'optimizer': opt_fn,
    })
    critic = Critic(ag, params={
        'input_layers': [cells],
        'NeuralNetworkModule': critic_net,
        'tau': TAU, 'tau_z': TAU_E, 'optimizer': opt_fn,
    })

    cfg = ExperimentConfig(
        label='train', n_episodes=N_EPISODES, t_timeout=15.0,
        eta=ETA, tau=TAU, tau_e=TAU_E,
    )
    for ep in range(N_EPISODES):
        _run_single_episode(env, ag, actor, critic, [cells], cfg)
        sf = np.mean(np.array(env.episodes['meta_info'][-100:]) == 'completed')
        if sf > 0.99 and ep > 10:
            break

    success_rate = np.mean(np.array(env.episodes['meta_info'][-100:]) == 'completed')
    return critic, actor, success_rate


def collect_data(ag, cells, critic_fta, fta_mod,
                 critic_base, relu1_base, n_steps):
    """Random exploration collecting raw cells + FTA/baseline representations.

    Returns (positions, raw, fta220, base_r1) as numpy arrays.
    """
    critic_fta.eval()
    critic_base.eval()

    positions = []
    raw_data = []
    fta220_acts = []   # 220-dim FTA output
    base_r1_acts = []  # 20-dim baseline ReLU1

    def fta220_hook(m, inp, out):
        fta220_acts.append(out.detach().cpu().numpy().flatten())

    def base_r1_hook(m, inp, out):
        base_r1_acts.append(out.detach().cpu().numpy().flatten())

    h_fta220 = fta_mod.register_forward_hook(fta220_hook)
    h_b_r1 = relu1_base.register_forward_hook(base_r1_hook)

    for _ in range(n_steps):
        ag.update()
        cells.update()

        positions.append(ag.pos.copy())
        fr = np.asarray(cells.get_state()).flatten()
        raw_data.append(fr.copy())

        state = torch.tensor(fr, dtype=torch.float).unsqueeze(0)
        with torch.inference_mode():
            critic_fta(state)
            critic_base(state)

    h_fta220.remove()
    h_b_r1.remove()

    critic_fta.train()
    critic_base.train()

    return (
        np.array(positions),
        np.array(raw_data),
        np.array(fta220_acts),
        np.array(base_r1_acts),
    )


def train_decoder(X, pos):
    """Train Ridge decoder, return model."""
    idx = np.arange(0, len(X), 5)
    X_sub, pos_sub = X[idx], pos[idx]
    lr = Ridge(alpha=0.01)
    lr.fit(X_sub, pos_sub)
    return lr


def decode_error(model, X_test, pos_test):
    """Mean Euclidean decoding error in cm."""
    pred = model.predict(X_test)
    return 100 * np.linalg.norm(pred - pos_test, axis=1).mean()


def bootstrap_ci(data, n_boot=10000, ci=95):
    """Bootstrap 95% CI for the mean."""
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lo, hi


# ── Main loop ────────────────────────────────────────────────────────────

def main():
    print(f"Cell type: {CELL_TYPE}")
    print(f"N_features: {N_FEATURES}, seeds: {N_REPEATS}")

    # results[rep_type, n_idx, repeat]  — LR decoding error only
    # rep_type: 0=raw, 1=fta, 2=relu
    # success_rates[agent_type, n_idx, repeat]
    # agent_type: 0=fta, 1=baseline
    checkpoint_path = os.path.join(OUT_DIR, f'decoding_{CELL_TYPE}_checkpoint.npz')
    if os.path.exists(checkpoint_path):
        ckpt = np.load(checkpoint_path)
        results = ckpt['results']
        success_rates = ckpt['success_rates']
        if results.shape == (3, len(N_FEATURES), N_REPEATS):
            print(f"Resumed from checkpoint ({np.count_nonzero(results)} nonzero entries)")
        else:
            print(f"Old checkpoint shape {results.shape}, starting fresh")
            results = np.zeros((3, len(N_FEATURES), N_REPEATS))
            success_rates = np.zeros((2, len(N_FEATURES), N_REPEATS))
    else:
        results = np.zeros((3, len(N_FEATURES), N_REPEATS))
        success_rates = np.zeros((2, len(N_FEATURES), N_REPEATS))

    train_steps = int(TRAIN_DURATION / DT)
    test_steps = int(TEST_DURATION / DT)

    for i, N in enumerate(tqdm(N_FEATURES, desc=f'{CELL_TYPE}')):
        for j in range(N_REPEATS):
            # Skip if already completed
            if results[0, i, j] != 0:
                print(f"\n  N={N}, seed={j+1}/{N_REPEATS} — already done, skipping")
                continue

            cell_seed = 1000 * i + j
            print(f"\n  N={N}, seed={j+1}/{N_REPEATS}, cell_seed={cell_seed}")

            # ── Train FTA agent ──────────────────────────────────────
            env_f, ag_f = make_task_env()
            np.random.seed(cell_seed)
            cells_f = make_cells(ag_f, CELL_TYPE, N)
            critic_fta_net, fta_mod = make_fta_critic(N)
            print(f"    Training FTA critic...")
            critic_f, actor_f, sf_f = train_agent(env_f, ag_f, cells_f, critic_fta_net, N)
            success_rates[0, i, j] = sf_f
            print(f"    FTA success rate: {sf_f:.2f}")

            # ── Train baseline agent ─────────────────────────────────
            env_b, ag_b = make_task_env()
            np.random.seed(cell_seed)
            cells_b = make_cells(ag_b, CELL_TYPE, N)
            critic_base_net, relu1_base = make_baseline_critic(N)
            print(f"    Training baseline critic...")
            critic_b, actor_b, sf_b = train_agent(env_b, ag_b, cells_b, critic_base_net, N)
            success_rates[1, i, j] = sf_b
            print(f"    Baseline success rate: {sf_b:.2f}")

            # ── Collect data (random exploration, shared trajectory) ─
            env_e = Environment()
            env_e.add_wall(WALL)
            ag_e = Agent(env_e, params={'dt': DT})
            np.random.seed(cell_seed)
            cells_e = make_cells(ag_e, CELL_TYPE, N)

            print(f"    Collecting training data ({TRAIN_DURATION/60:.0f} min)...")
            pos_train, raw_train, fta_train, relu_train = collect_data(
                ag_e, cells_e, critic_fta_net, fta_mod,
                critic_base_net, relu1_base, train_steps,
            )

            # ── Train decoders ───────────────────────────────────────
            print(f"    Training decoders... (raw:{raw_train.shape[1]}, "
                  f"fta:{fta_train.shape[1]}, relu:{relu_train.shape[1]})")
            lr_raw = train_decoder(raw_train, pos_train)
            lr_fta = train_decoder(fta_train, pos_train)
            lr_relu = train_decoder(relu_train, pos_train)

            # ── Collect test data ────────────────────────────────────
            print(f"    Collecting test data ({TEST_DURATION/60:.0f} min)...")
            pos_test, raw_test, fta_test, relu_test = collect_data(
                ag_e, cells_e, critic_fta_net, fta_mod,
                critic_base_net, relu1_base, test_steps,
            )

            # ── Compute errors ───────────────────────────────────────
            results[0, i, j] = decode_error(lr_raw, raw_test, pos_test)
            results[1, i, j] = decode_error(lr_fta, fta_test, pos_test)
            results[2, i, j] = decode_error(lr_relu, relu_test, pos_test)

            print(f"    LR errors — raw: {results[0,i,j]:.1f}, "
                  f"fta: {results[1,i,j]:.1f}, relu: {results[2,i,j]:.1f} cm")

            # Checkpoint after every iteration
            np.savez(checkpoint_path, results=results, success_rates=success_rates)

    # ── Save results ─────────────────────────────────────────────────────
    npz_path = os.path.join(OUT_DIR, f'decoding_{CELL_TYPE}_results.npz')
    np.savez(npz_path, results=results, success_rates=success_rates)
    print(f"\nSaved results → {npz_path}")

    # ── Plot ─────────────────────────────────────────────────────────────
    cell_label = {'place': 'Place cells', 'grid': 'Grid cells', 'bvc': 'BV cells'}[CELL_TYPE]

    fig, ax = plt.subplots(figsize=(6, 4))

    for rep_idx, color, label in [
        (0, 'C1', f'Raw {cell_label.lower()}'),
        (1, 'C0', 'FTA'),
        (2, 'C2', 'ReLU'),
    ]:
        means, ci_lo, ci_hi = [], [], []
        for i in range(len(N_FEATURES)):
            vals = results[rep_idx, i, :]
            m = np.mean(vals)
            lo, hi = bootstrap_ci(vals)
            means.append(m)
            ci_lo.append(lo)
            ci_hi.append(hi)
        means = np.array(means)
        ci_lo = np.array(ci_lo)
        ci_hi = np.array(ci_hi)

        ax.scatter(N_FEATURES, means, c=color, zorder=3)
        ax.plot(N_FEATURES, means, c=color, label=label, linewidth=1)
        ax.fill_between(N_FEATURES, ci_lo, ci_hi, facecolor=color, alpha=0.3)

    ax.set_xlabel('Number of input cells (log scale)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(axis='y', which='minor', left=False)
    ax.set_xbound(lower=N_FEATURES[-1] * 0.8, upper=N_FEATURES[0] / 0.8)
    ax.set_ylabel('Average decoding error (log scale)')
    ax.set_title(f'{cell_label} — Linear Decodeability')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(N_FEATURES)
    ax.set_xticklabels(N_FEATURES)
    log2_cms = np.logspace(0, 4, 5, base=2, dtype=int)
    ax.set_yticks(log2_cms)
    ax.set_yticklabels(log2_cms)
    ax.legend()
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f'decoding_{CELL_TYPE}_linear.pdf')
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved plot → {pdf_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
