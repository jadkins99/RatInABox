"""Configuration-driven experiment runner for actor-critic navigation tasks.

Defines an ``ExperimentConfig`` dataclass and a ``run_experiment`` function
that handles environment creation, network construction, training, and
plotting -- replacing the repeated setup/train/plot blocks in the notebook.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

import ratinabox
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells
from ratinabox.contribs.NeuralNetworkNeurons import NeuralNetworkNeurons
from ratinabox.contribs.TaskEnvironment import (
    SpatialGoalEnvironment,
    SpatialGoal,
    Reward,
)

from networks import (
    Backbone,
    VxVyGaussianHead,
    NESWCategoricalHead,
    make_mlp_critic,
    make_fta_critic,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """All settings needed to run one actor-critic navigation experiment."""

    label: str = "experiment"

    # Task constants
    dt: float = 0.1
    t_timeout: float = 15.0
    goal_pos: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))
    wall: object = None
    goal_radius: float = 0.1
    reward: float = 1.0
    reward_duration: float = 1.0

    # Learning constants
    tau: float = 5.0
    tau_e: float = 5.0
    eta: float = 0.01
    n_episodes: int = 5000
    l2: float = 0.0

    # Place cell settings
    n_place_cells: int = 50

    # Network type: "mlp" or "fta"
    critic_type: Literal["mlp", "fta"] = "mlp"
    actor_hidden: list = field(default_factory=lambda: [50])

    # MLP-specific
    critic_hidden: list = field(default_factory=lambda: [20, 20])

    # FTA-specific
    fta_eta: float | None = None  # None → eta=delta (tile width)
    fta_input_min: float = 0.0
    fta_input_max: float = 1.0
    fta_n_tiles: int = 10
    fta_n_tilings: int = 1
    fta_pre_fta: list = field(default_factory=lambda: [20])
    fta_post_fta: list = field(default_factory=lambda: [1])

    # Early stopping
    success_threshold: float = 0.99
    min_episodes: int = 10

    # Whether the actor uses egocentric actions
    egocentric_actions: bool = False


def _make_env_and_agent(cfg: ExperimentConfig):
    """Build a SpatialGoalEnvironment + Agent from *cfg*."""
    env = SpatialGoalEnvironment(
        dt=cfg.dt,
        teleport_on_reset=True,
        episode_terminate_delay=cfg.reward_duration,
    )
    env.exploration_strength = 1
    if cfg.wall is not None:
        env.add_wall(cfg.wall)

    reward = Reward(cfg.reward, decay="none", expire_clock=cfg.reward_duration, dt=cfg.dt)
    goals = [SpatialGoal(env, pos=cfg.goal_pos, goal_radius=cfg.goal_radius, reward=reward)]
    env.goal_cache.reset_goals = goals

    ag = Agent(env, params={"dt": cfg.dt})
    env.add_agents(ag)
    return env, ag


def _make_networks(cfg: ExperimentConfig, n_in: int):
    """Build critic and actor networks from *cfg*."""
    if cfg.critic_type == "fta":
        critic_nn = make_fta_critic(
            n_in=n_in,
            pre_fta=cfg.fta_pre_fta,
            post_fta=cfg.fta_post_fta,
            input_min=cfg.fta_input_min,
            input_max=cfg.fta_input_max,
            n_tiles=cfg.fta_n_tiles,
            n_tilings=cfg.fta_n_tilings,
            eta=cfg.fta_eta,
        )
    else:
        critic_nn = make_mlp_critic(n_in=n_in, hidden=cfg.critic_hidden)

    actor_nn = VxVyGaussianHead(
        Backbone(n_in=n_in, n_out=2, hidden=cfg.actor_hidden)
    )
    return critic_nn, actor_nn


def _ego_to_allo(v_ego, head_direction):
    """Convert egocentric velocity to allocentric.

    Note: also available as ``viz.ego_to_allo``.
    """
    bearing = ratinabox.utils.get_bearing(head_direction)
    return ratinabox.utils.rotate(v_ego, -bearing)


def _run_single_episode(env, ag, actor, critic, state_cells, cfg: ExperimentConfig):
    """Run one episode of the training loop."""
    critic.initialise_traces()
    actor.initialise_traces()

    while True:
        action, log_prob = actor.NeuralNetworkModule.sample_action(actor.firingrate_torch)
        if cfg.egocentric_actions:
            action = _ego_to_allo(action, ag.head_direction)

        _, reward, terminate_episode, _, _ = env.step1(
            action=action,
            drift_to_random_strength_ratio=1,
        )

        for cell in state_cells:
            cell.update()

        critic.update(reward=reward)
        actor.update(log_prob=log_prob, td_error=critic.td_error)

        if env.t - env.episodes["start"][-1] > cfg.t_timeout:
            env.reset(episode_meta_info="timeout")
            return
        elif terminate_episode:
            env.reset(episode_meta_info="completed")
            return


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Holds everything produced by a single experiment run."""
    env: object
    agent: object
    placecells: object
    actor: object
    critic: object
    config: ExperimentConfig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment(cfg: ExperimentConfig, ActorCls=None, CriticCls=None) -> ExperimentResult:
    """Run a full actor-critic experiment from *cfg*.

    ``ActorCls`` and ``CriticCls`` should be the ``Actor`` and ``Critic``
    classes (imported from the notebook / ac module).  They default to
    being imported lazily so this module has no circular dependency.

    Returns an ``ExperimentResult`` with all the objects you need for
    further analysis or plotting.
    """
    # Lazy import fallback -- caller should pass these explicitly
    if ActorCls is None or CriticCls is None:
        raise ValueError("ActorCls and CriticCls must be provided")

    # 1. Environment + agent
    env, ag = _make_env_and_agent(cfg)
    placecells = PlaceCells(ag, params={"n": cfg.n_place_cells})

    # 2. Networks
    critic_nn, actor_nn = _make_networks(cfg, n_in=placecells.n)

    # 3. Actor / Critic wrappers
    optimizer_fn = lambda params: torch.optim.SGD(
        params, lr=cfg.eta, maximize=True, weight_decay=cfg.l2
    )
    actor = ActorCls(
        ag,
        params={
            "input_layers": [placecells],
            "NeuralNetworkModule": actor_nn,
            "tau": cfg.tau,
            "tau_z": cfg.tau_e,
            "optimizer": optimizer_fn,
        },
    )
    actor.colormap = "PiYG"

    critic = CriticCls(
        ag,
        params={
            "input_layers": [placecells],
            "NeuralNetworkModule": critic_nn,
            "tau": cfg.tau,
            "tau_z": cfg.tau_e,
            "optimizer": optimizer_fn,
        },
    )

    # 4. Training loop
    try:
        for i in (pbar := tqdm(range(cfg.n_episodes), desc=cfg.label)):
            _run_single_episode(env, ag, actor, critic, [placecells], cfg)
            success_frac = np.mean(
                np.array(env.episodes["meta_info"][-100:]) == "completed"
            )
            episode_time = np.mean(env.episodes["duration"][-100:])
            pbar.set_description(
                f"{cfg.label} | success: {success_frac:.2f}, time: {episode_time:.1f}"
            )
            if success_frac > cfg.success_threshold and i > cfg.min_episodes:
                break
    except KeyboardInterrupt:
        print("Interrupted by user")

    return ExperimentResult(
        env=env,
        agent=ag,
        placecells=placecells,
        actor=actor,
        critic=critic,
        config=cfg,
    )
