"""Visualization helpers for actor-critic navigation experiments."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import ratinabox


def display_reward_patch(
    fig,
    ax,
    reward_pos=np.array([0.5, 0.5]),
    reward_radius=0.1,
    **kwargs,
):
    """Draw a translucent red circle showing the reward zone."""
    circle = matplotlib.patches.Circle(
        reward_pos, radius=reward_radius, facecolor="r", alpha=0.2, color=None
    )
    ax.add_patch(circle)
    return fig, ax


def plot_reward_history(env, smooth=100):
    """Plot episode durations with a smoothed overlay."""
    fig, ax = plt.subplots(figsize=(6, 2))
    data = env.episodes
    episodes = data["episode"][:-1]
    durations = data["duration"]
    smoothed = np.convolve(durations, np.ones(smooth) / smooth, mode="full")[
        : len(durations)
    ]
    ax.scatter(episodes, durations, alpha=0.4)
    ax.plot(episodes[smooth:], smoothed[smooth:], color="red")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Duration")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(durations) + 0.1)
    return fig, ax


def ego_to_allo(v_ego, head_direction):
    """Rotate an egocentric velocity vector into allocentric coordinates."""
    bearing = ratinabox.utils.get_bearing(head_direction)
    return ratinabox.utils.rotate(v_ego, -bearing)


def plot_experiment_results(result, display_reward_patch_fn=None):
    """Standard post-training plots for an ExperimentResult.

    Plots: reward history, value function, policy, and recent trajectory.
    """
    env = result.env
    ag = result.agent

    fig, ax = plot_reward_history(env)
    fig, ax = result.critic.plot_rate_map()
    fig.suptitle(f"{result.config.label} — Value function (after learning)")
    fig, ax = result.actor.plot_rate_map(zero_center=True)
    fig.suptitle(f"{result.config.label} — Policy (after learning)")
    ax[0].set_title("Vx")
    ax[1].set_title("Vy")

    fig, ax = ag.plot_trajectory(
        color="changing",
        t_start=env.episodes["start"][-100],
        t_end=env.episodes["start"][-1],
    )
    if display_reward_patch_fn is not None:
        display_reward_patch_fn(fig, ax)
    else:
        display_reward_patch(fig, ax, result.config.goal_pos, result.config.goal_radius)
    return fig, ax
