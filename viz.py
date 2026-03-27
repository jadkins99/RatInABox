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


def sparsity_metric(activation: np.ndarray, threshold: float = 0.01) -> float:
    """Fraction of activation mass above *threshold*."""
    above = activation[activation > threshold]
    return float(np.sum(above) / activation.size) if activation.size > 0 else 0.0


def compute_sparsity_map(activation_dict: dict, threshold: float = 0.01) -> dict:
    """Compute per-location mean sparsity from an ActivationRecorder store.

    Args:
        activation_dict: ``{obs_key: [tensor, ...]}`` from ``recorder.get(name)``.
        threshold: passed to :func:`sparsity_metric`.

    Returns:
        ``{obs_key: float}`` mapping each observation to its mean sparsity.
    """
    sparsity_dict = {}
    for location, tensors in activation_dict.items():
        vals = [sparsity_metric(t.numpy(), threshold) for t in tensors]
        sparsity_dict[location] = sum(vals) / len(vals)
    return sparsity_dict


def plot_sparsity_map(
    env,
    sparsity_dictionary: dict,
    bins: int = 60,
    fig=None,
    ax=None,
    colorbar: bool = True,
    title: str = "Sparsity heatmap",
    autosave=None,
    **kwargs,
):
    """Plot a heatmap of a scalar sparsity metric over space.

    Args:
        env: a RatInABox environment (2-D or 1-D).
        sparsity_dictionary: ``{(x, y): scalar}`` from :func:`compute_sparsity_map`.
        bins: resolution of the spatial grid.
    """
    dim = env.dimensionality

    xs, ys, vs = [], [], []
    for loc, v in sparsity_dictionary.items():
        if dim == "2D":
            if isinstance(loc, tuple) and len(loc) == 2:
                x, y = loc
            else:
                ll = tuple(loc)
                x, y = ll[0], ll[1]
            xs.append(float(x))
            ys.append(float(y))
            vs.append(float(v))
        else:
            x = loc[0] if isinstance(loc, tuple) else loc
            xs.append(float(x))
            vs.append(float(v))

    xs = np.asarray(xs)
    vs = np.asarray(vs)

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(4, 4) if dim == "2D" else (6, 2))
    fig, ax = env.plot_environment(fig=fig, ax=ax, autosave=False, **kwargs)

    if dim == "2D":
        ys = np.asarray(ys)
        ex = env.extent
        xmin, xmax, ymin, ymax = ex

        x_edges = np.linspace(xmin, xmax, bins + 1)
        y_edges = np.linspace(ymin, ymax, bins + 1)

        sum_grid, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges], weights=vs)
        cnt_grid, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])

        mean_grid = sum_grid / np.maximum(cnt_grid, 1)
        mean_grid[cnt_grid == 0] = np.nan

        im = ax.imshow(
            mean_grid.T,
            origin="lower",
            extent=ex,
            interpolation="nearest",
            zorder=0,
            vmin=0,
            vmax=1.0,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(length=0)
            cbar.set_label("sparsity metric", labelpad=10)
            cbar.outline.set_visible(False)
    else:
        ex = env.extent
        xmin, xmax = ex
        x_edges = np.linspace(xmin, xmax, bins + 1)
        sum_bins, _ = np.histogram(xs, bins=x_edges, weights=vs)
        cnt_bins, _ = np.histogram(xs, bins=x_edges)
        mean_bins = sum_bins / np.maximum(cnt_bins, 1)
        mean_bins[cnt_bins == 0] = np.nan
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        ax.plot(x_centers, mean_bins, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Position / m")
        ax.set_ylabel("sparsity metric")

    if autosave is not None:
        ratinabox.utils.save_figure(fig, f"sparsitymap_{title}", save=autosave)
    return fig, ax
