"""Decoding position from raw cells vs learned FTA/baseline representations.

Usage: python run_decoding.py <cell_type>
  cell_type: 'place', 'grid', or 'bvc'

For each number-of-cells N, trains an FTA critic and a baseline critic on
the goal-reaching task, then measures how well position can be decoded
(Ridge + GP regression) from:
  1. Raw cell firing rates (N-dim)
  2. FTA 220-dim (post-FTA, pre-compression)
  3. FTA 20-dim (post-compression ReLU)
  4. Baseline 20-dim (ReLU1, after first linear)
  5. Baseline 20-dim (ReLU2, after second linear)

Outputs:
  decoding_<cell_type>_results.npy  — results array
  decoding_<cell_type>.pdf          — GP and LR decoding-error-vs-N plots
"""

import sys, os, importlib
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

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
# Auto-detect the installed fta package location to avoid hardcoded paths.
# We use importlib to load fta.torch directly (bypassing the local fta.py).
import importlib.metadata as _meta
_fta_dist = _meta.distribution('fuzzy-tiling-activation')
_fta_pkg_dir = str(_fta_dist._path.parent)  # site-packages directory
_fta_torch_path = os.path.join(_fta_pkg_dir, 'fta', 'torch.py')
spec = importlib.util.spec_from_file_location('fta_pypi_torch', _fta_torch_path)
fta_pypi_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fta_pypi_mod)
PyPiFTA = fta_pypi_mod.FTA

# ── Constants ────────────────────────────────────────────────────────────
CELL_TYPE = sys.argv[1] if len(sys.argv) > 1 else 'place'
N_FEATURES = [80, 40, 20, 10]
N_REPEATS = 5
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
ETA = 0.002  # reduced from 0.01 to compensate for accumulating trace (factor of tau_z=5)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Helpers ──────────────────────────────────────────────────────────────


def make_cells(agent, cell_type, n):
    if cell_type == 'place':
        return PlaceCells(agent, params={
            'n': n, 'description': 'gaussian_threshold', 'widths': 0.4,
        })
    elif cell_type == 'grid':
        return GridCells(agent, params={'n': n, 'gridscale': 0.4})
    elif cell_type == 'bvc':
        return BoundaryVectorCells(agent, params={'n': n})
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")


def make_task_env():
    """Create a SpatialGoalEnvironment with the wall for RL training."""
    env = SpatialGoalEnvironment(
        dt=DT, teleport_on_reset=True, episode_terminate_delay=1.0,
    )
    env.exploration_strength = 1
    env.add_wall(WALL)
    reward = Reward(1.0, decay='none', expire_clock=1.0, dt=DT)
    goals = [SpatialGoal(env, pos=GOAL_POS, goal_radius=GOAL_RADIUS, reward=reward)]
    env.goal_cache.reset_goals = goals
    ag = Agent(env, params={'dt': DT})
    env.add_agents(ag)
    return env, ag


def make_fta_critic(n_in):
    """Linear(n_in→20) → LN (non-adaptive) → FTA → Linear(220→20) → ReLU → Linear(20→1)
    Returns (net, fta_module, relu_module)."""
    fta = PyPiFTA(
        bound=BOUND, spillover_base=0,
        spillover_mode='derive_from_tile_width',
        tile_width=None, num_tiles=N_TILES,
    )
    total_tiles = fta.num_tiles  # 11
    fta_out_dim = PRE_FTA_DIM * total_tiles  # 220
    relu = nn.ReLU()
    net = nn.Sequential(
        nn.Linear(n_in, PRE_FTA_DIM),
        nn.LayerNorm(PRE_FTA_DIM, elementwise_affine=False),
        fta,
        nn.Linear(fta_out_dim, PRE_FTA_DIM),
        relu,
        nn.Linear(PRE_FTA_DIM, 1),
    )
    return net, fta, relu


def make_baseline_critic(n_in):
    """Linear(n_in→20) → ReLU1 → Linear(20→20) → ReLU2 → Linear(20→1)
    Returns (net, relu1_module, relu2_module)."""
    relu1 = nn.ReLU()
    relu2 = nn.ReLU()
    net = nn.Sequential(
        nn.Linear(n_in, PRE_FTA_DIM),
        relu1,
        nn.Linear(PRE_FTA_DIM, PRE_FTA_DIM),
        relu2,
        nn.Linear(PRE_FTA_DIM, 1),
    )
    return net, relu1, relu2


def train_agent(env, ag, cells, critic_net, n_in):
    """Train an actor-critic agent for N_EPISODES, return (critic, actor)."""
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
    return critic, actor


def collect_data(ag, cells, critic_fta, fta_mod, relu_fta,
                 critic_base, relu1_base, relu2_base, n_steps):
    """Random exploration collecting raw cells + FTA/baseline representations.

    All five representations are recorded on the exact same trajectory.
    Returns (positions, raw, fta220, fta20, base_r1, base_r2) as numpy arrays.
    """
    critic_fta.eval()
    critic_base.eval()

    positions = []
    raw_data = []
    fta220_acts = []   # 220-dim FTA output
    fta20_acts = []    # 20-dim post-compression ReLU
    base_r1_acts = []  # 20-dim baseline ReLU1
    base_r2_acts = []  # 20-dim baseline ReLU2

    def fta220_hook(m, inp, out):
        fta220_acts.append(out.detach().cpu().numpy().flatten())

    def fta20_hook(m, inp, out):
        fta20_acts.append(out.detach().cpu().numpy().flatten())

    def base_r1_hook(m, inp, out):
        base_r1_acts.append(out.detach().cpu().numpy().flatten())

    def base_r2_hook(m, inp, out):
        base_r2_acts.append(out.detach().cpu().numpy().flatten())

    h_fta220 = fta_mod.register_forward_hook(fta220_hook)
    h_fta20 = relu_fta.register_forward_hook(fta20_hook)
    h_b_r1 = relu1_base.register_forward_hook(base_r1_hook)
    h_b_r2 = relu2_base.register_forward_hook(base_r2_hook)

    for _ in range(n_steps):
        ag.update()
        cells.update()

        positions.append(ag.pos.copy())
        fr = np.asarray(cells.get_state()).flatten()
        raw_data.append(fr.copy())

        state = torch.tensor(fr, dtype=torch.float).unsqueeze(0)  # (1, N)
        with torch.inference_mode():
            critic_fta(state)
            critic_base(state)

    h_fta220.remove()
    h_fta20.remove()
    h_b_r1.remove()
    h_b_r2.remove()

    critic_fta.train()
    critic_base.train()

    return (
        np.array(positions),
        np.array(raw_data),
        np.array(fta220_acts),
        np.array(fta20_acts),
        np.array(base_r1_acts),
        np.array(base_r2_acts),
    )


def train_decoders(X, pos):
    """Train GP and Ridge decoders, return (gp_model, lr_model)."""
    # Subsample every 5th step for efficiency
    idx = np.arange(0, len(X), 5)
    X_sub, pos_sub = X[idx], pos[idx]

    dim = X.shape[1]
    gp = GaussianProcessRegressor(
        alpha=0.01,
        kernel=RBF(np.sqrt(dim / 20), length_scale_bounds='fixed'),
    )
    lr = Ridge(alpha=0.01)
    gp.fit(X_sub, pos_sub)
    lr.fit(X_sub, pos_sub)
    return gp, lr


def decode_error(model, X_test, pos_test):
    """Mean Euclidean decoding error in cm."""
    pred = model.predict(X_test)
    return 100 * np.linalg.norm(pred - pos_test, axis=1).mean()


# ── Main loop ────────────────────────────────────────────────────────────

def main():
    print(f"Cell type: {CELL_TYPE}")
    print(f"N_features: {N_FEATURES}, repeats: {N_REPEATS}")

    # results[rep_type, n_idx, repeat, decoder_type]
    # rep_type: 0=raw, 1=fta220, 2=fta20, 3=base_r1, 4=base_r2
    # decoder_type: 0=GP, 1=LR
    checkpoint_path = os.path.join(OUT_DIR, f'decoding_{CELL_TYPE}_checkpoint.npy')
    if os.path.exists(checkpoint_path):
        results = np.load(checkpoint_path)
        if results.shape[0] == 5:
            print(f"Resumed from checkpoint ({np.count_nonzero(results)} nonzero entries)")
        else:
            print(f"Old checkpoint shape {results.shape}, starting fresh")
            results = np.zeros((5, len(N_FEATURES), N_REPEATS, 2))
    else:
        results = np.zeros((5, len(N_FEATURES), N_REPEATS, 2))

    train_steps = int(TRAIN_DURATION / DT)
    test_steps = int(TEST_DURATION / DT)

    for i, N in enumerate(tqdm(N_FEATURES, desc=f'{CELL_TYPE}')):
        for j in range(N_REPEATS):
            # Skip if already completed (nonzero results for this slot)
            if results[0, i, j, 0] != 0 or results[0, i, j, 1] != 0:
                print(f"\n  N={N}, repeat={j+1}/{N_REPEATS} — already done, skipping")
                continue

            cell_seed = 1000 * i + j
            print(f"\n  N={N}, repeat={j+1}/{N_REPEATS}, seed={cell_seed}")

            # ── Train FTA agent ──────────────────────────────────────
            env_f, ag_f = make_task_env()
            np.random.seed(cell_seed)
            cells_f = make_cells(ag_f, CELL_TYPE, N)
            critic_fta_net, fta_mod, relu_fta = make_fta_critic(N)
            print(f"    Training FTA critic...")
            critic_f, actor_f = train_agent(env_f, ag_f, cells_f, critic_fta_net, N)
            sf_f = np.mean(np.array(env_f.episodes['meta_info'][-100:]) == 'completed')
            print(f"    FTA success rate: {sf_f:.2f}")

            # ── Train baseline agent ─────────────────────────────────
            env_b, ag_b = make_task_env()
            np.random.seed(cell_seed)
            cells_b = make_cells(ag_b, CELL_TYPE, N)
            critic_base_net, relu1_base, relu2_base = make_baseline_critic(N)
            print(f"    Training baseline critic...")
            critic_b, actor_b = train_agent(env_b, ag_b, cells_b, critic_base_net, N)
            sf_b = np.mean(np.array(env_b.episodes['meta_info'][-100:]) == 'completed')
            print(f"    Baseline success rate: {sf_b:.2f}")

            # ── Collect data (random exploration, shared trajectory) ─
            env_e = Environment()
            env_e.add_wall(WALL)
            ag_e = Agent(env_e, params={'dt': DT})
            np.random.seed(cell_seed)
            cells_e = make_cells(ag_e, CELL_TYPE, N)

            print(f"    Collecting training data ({TRAIN_DURATION/60:.0f} min)...")
            pos_train, raw_train, fta220_train, fta20_train, base_r1_train, base_r2_train = collect_data(
                ag_e, cells_e, critic_fta_net, fta_mod, relu_fta,
                critic_base_net, relu1_base, relu2_base, train_steps,
            )

            # ── Train decoders ───────────────────────────────────────
            print(f"    Training decoders... (raw:{raw_train.shape[1]}, "
                  f"fta220:{fta220_train.shape[1]}, fta20:{fta20_train.shape[1]}, "
                  f"base_r1:{base_r1_train.shape[1]}, base_r2:{base_r2_train.shape[1]})")
            gp_raw, lr_raw = train_decoders(raw_train, pos_train)
            gp_fta220, lr_fta220 = train_decoders(fta220_train, pos_train)
            gp_fta20, lr_fta20 = train_decoders(fta20_train, pos_train)
            gp_base_r1, lr_base_r1 = train_decoders(base_r1_train, pos_train)
            gp_base_r2, lr_base_r2 = train_decoders(base_r2_train, pos_train)

            # ── Collect test data ────────────────────────────────────
            print(f"    Collecting test data ({TEST_DURATION/60:.0f} min)...")
            pos_test, raw_test, fta220_test, fta20_test, base_r1_test, base_r2_test = collect_data(
                ag_e, cells_e, critic_fta_net, fta_mod, relu_fta,
                critic_base_net, relu1_base, relu2_base, test_steps,
            )

            # ── Compute errors ───────────────────────────────────────
            results[0, i, j, 0] = decode_error(gp_raw, raw_test, pos_test)
            results[1, i, j, 0] = decode_error(gp_fta220, fta220_test, pos_test)
            results[2, i, j, 0] = decode_error(gp_fta20, fta20_test, pos_test)
            results[3, i, j, 0] = decode_error(gp_base_r1, base_r1_test, pos_test)
            results[4, i, j, 0] = decode_error(gp_base_r2, base_r2_test, pos_test)
            results[0, i, j, 1] = decode_error(lr_raw, raw_test, pos_test)
            results[1, i, j, 1] = decode_error(lr_fta220, fta220_test, pos_test)
            results[2, i, j, 1] = decode_error(lr_fta20, fta20_test, pos_test)
            results[3, i, j, 1] = decode_error(lr_base_r1, base_r1_test, pos_test)
            results[4, i, j, 1] = decode_error(lr_base_r2, base_r2_test, pos_test)

            print(f"    GP errors  — raw: {results[0,i,j,0]:.1f}, "
                  f"fta220: {results[1,i,j,0]:.1f}, fta20: {results[2,i,j,0]:.1f}, "
                  f"base_r1: {results[3,i,j,0]:.1f}, base_r2: {results[4,i,j,0]:.1f} cm")
            print(f"    LR errors  — raw: {results[0,i,j,1]:.1f}, "
                  f"fta220: {results[1,i,j,1]:.1f}, fta20: {results[2,i,j,1]:.1f}, "
                  f"base_r1: {results[3,i,j,1]:.1f}, base_r2: {results[4,i,j,1]:.1f} cm")

            # Checkpoint after every iteration
            np.save(checkpoint_path, results)

    # ── Save results ─────────────────────────────────────────────────────
    npy_path = os.path.join(OUT_DIR, f'decoding_{CELL_TYPE}_results.npy')
    np.save(npy_path, results)
    print(f"\nSaved results → {npy_path}")

    # ── Plot ─────────────────────────────────────────────────────────────
    cell_label = {'place': 'Place cells', 'grid': 'Grid cells', 'bvc': 'BV cells'}[CELL_TYPE]
    plot_colors = ['C1', 'C0', 'C4', 'C2', 'C3']
    plot_labels = [
        f'Raw {cell_label.lower()}',
        'FTA 220-dim (post-FTA)',
        'FTA 20-dim (post-ReLU)',
        'Baseline 20-dim (ReLU1)',
        'Baseline 20-dim (ReLU2)',
    ]

    for dec_idx, dec_name in enumerate(['Gaussian process regression', 'Linear ridge regression']):
        means = np.mean(results[:, :, :, dec_idx], axis=2)
        sems = np.std(results[:, :, :, dec_idx], axis=2) / np.sqrt(N_REPEATS)

        fig, ax = plt.subplots(figsize=(6, 4))
        for k, (color, label) in enumerate(zip(plot_colors, plot_labels)):
            ax.scatter(N_FEATURES, means[k], c=color, zorder=3)
            ax.plot(N_FEATURES, means[k], c=color, label=label, linewidth=1)
            ax.fill_between(
                N_FEATURES,
                means[k] - sems[k], means[k] + sems[k],
                facecolor=color, alpha=0.3,
            )

        ax.set_xlabel('Number of input cells (log scale)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='x', which='minor', bottom=False)
        ax.tick_params(axis='y', which='minor', left=False)
        ax.set_xbound(lower=N_FEATURES[-1] * 0.8, upper=N_FEATURES[0] / 0.8)
        ax.set_ylabel('Average decoding error / cm (log scale)')
        ax.set_title(f'{cell_label} — {dec_name}')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(N_FEATURES)
        ax.set_xticklabels(N_FEATURES)
        log2_cms = np.logspace(0, 4, 5, base=2, dtype=int)
        ax.set_yticks(log2_cms)
        ax.set_yticklabels(log2_cms)
        ax.legend()
        fig.tight_layout()

        suffix = 'gp' if dec_idx == 0 else 'lr'
        pdf_path = os.path.join(OUT_DIR, f'decoding_{CELL_TYPE}_{suffix}.pdf')
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"Saved plot → {pdf_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
