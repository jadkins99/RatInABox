import os
import matplotlib 
import matplotlib.pyplot as plt 
import numpy as np
import ratinabox

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

def display_reward_patch(fig, ax, reward_pos=np.array([0.5, 0.5]), reward_radius=0.1, **kwargs): #we'll also use this later 
    """Plots the reward patch on the given axis"""
    circle = matplotlib.patches.Circle(reward_pos, radius=reward_radius,
                                       facecolor='r', alpha=0.2, color=None) 
    ax.add_patch(circle)
    return fig, ax 


def plot_reward_history(env, smooth=100):
    """Takes an environment and used its episode data to diplay the time series of rewards and the same data smoothed over `smooth` episodes"""
    fig, ax = plt.subplots(figsize=(6,2))
    data = env.episodes
    episodes = data['episode'][:-1]
    durations = data['duration']
    smoothed_durations = np.convolve(durations, np.ones(smooth)/smooth, mode='full')[:len(durations)]
    ax.scatter(episodes,durations,alpha=0.4)
    ax.plot(episodes[smooth:],smoothed_durations[smooth:],color='red')
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Duration")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0,max(durations)+0.1)
    return fig, ax

def ego_to_allo(v_ego, head_direction):
    """Converts an egocentric velocity vector to an allocentric one by by rotating it by the bearing of the agents current head direction"""
    bearing = ratinabox.utils.get_bearing(head_direction)
    v_allo = ratinabox.utils.rotate(v_ego, -bearing) #bearing measured clockwise from north, so we rotate anticlockwise by -bearing
    return v_allo

def plot_rate_maps(env, ag, placecells, actor, critic, goal_pos, reward_radius, time, reward=False, trajectory=False, save_dir='figures'):
    
    os.makedirs(save_dir, exist_ok=True)

    if reward:
        fig, ax = plot_reward_history(env)
        fig.savefig(os.path.join(save_dir, "reward_history.png"))

    fig, ax = critic.plot_rate_map()
    fig.suptitle("Value function")
    fig.savefig(os.path.join(save_dir, f"value_function_{time}.png"))

    fig, ax = actor.plot_rate_map(zero_center=True)
    fig.suptitle("Policy")
    ax[0].set_title("Vx")
    ax[1].set_title("Vy")
    fig.savefig(os.path.join(save_dir, f"policy_{time}.png"))

    fig, ax = placecells.plot_place_cell_locations()
    fig.savefig(os.path.join(save_dir, f"place_cell_locations_{time}.png"))

    if trajectory:
        fig, ax = ag.plot_trajectory(
            color="changing",
            t_start=env.episodes['start'][0],
            t_end=env.episodes['start'][0]
        )
        display_reward_patch(fig, ax, reward_pos=goal_pos, reward_radius=reward_radius)
        fig.savefig(os.path.join(save_dir, "trajectory.png"))


def plot_neurons_over_time(
    x,
    y,
    x_label,
    y_label,
    se=None,          
    save=False,
    filename=None
):
    """
    Plot dead neurons over time with optional standard error shading.

    Args:
        x: x-axis values
        y: mean values
        x_label: label for x-axis
        y_label: label for y-axis
        se: standard error (same shape as y)
        save: whether to save the figure
        filename: path to save the figure
    """

    plt.figure(figsize=(8, 4))

    # Main line
    plt.plot(x, y, lw=2)

    # Add standard error shading
    if se is not None:
        y = np.array(y)
        se = np.array(se)

        plt.fill_between(
            x,
            y - se,
            y + se,
            alpha=0.3
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Remove grid
    plt.grid(False)

    # Remove top and right borders
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save:
        if filename is None:
            raise ValueError("filename must be provided when save=True")
        plt.savefig(filename, dpi=300, bbox_inches="tight")


def get_timesteps_for_percentages(total_timesteps, percentages=[0.1,0.3,0.5,0.7,0.9,1.0]):
    """
    Returns the actual timestep indices corresponding to percentages of total training.
    """
    timestep_indices = {}
    for p in percentages:
        idx = int(p * (total_timesteps - 1))  # 0-based indexing
        timestep_indices[int(p*100)] = idx
    return timestep_indices

def plot_bin_counts_per_percentage(bin_counts, percentages=[10,30,50,70,90,100], save=False, filename=None):
    """
    Plots the bin counts for each percentage of training.
    """
    timesteps = bin_counts.shape[0]
    
    # Get timestep indices
    timestep_indices = get_timesteps_for_percentages(timesteps, [p/100 for p in percentages])

    for perc, idx in timestep_indices.items():
        plt.figure(figsize=(6,4))
        plt.bar(np.arange(bin_counts.shape[1]), bin_counts[idx])
        plt.xlabel("Bins")
        plt.ylabel("Active neuron count")
        plt.title(f"Active neurons per bin at {perc}% of training (timestep {idx})")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(False)
        if save:
            plt.savefig(f"{filename}/bin_counts_{perc}percent.png", dpi=300, bbox_inches="tight")



def plot_occupancy_map(
    occupancy_map,
    save_dir=None,
    filename=None,
    vmin=0,   
    vmax=2    
):
    """
    Plot the occupancy map (proportion of time spent in each bin).
    Unvisited bins are set to the minimum visited value.

    Args:
        occupancy_map: np.array of shape (n_bins, n_bins)
        save_dir:      directory to save the plot (optional)
        filename:      file name for saving (optional)
        vmin:          minimum value for color scale (optional)
        vmax:          maximum value for color scale (optional)
    """

    # Fill NaNs with minimum visited value (for visualization only)
    min_val = np.nanmin(occupancy_map)
    occupancy_filled = np.where(np.isnan(occupancy_map), min_val, occupancy_map)

    # If vmin/vmax not provided, default to data range

    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(
        occupancy_filled.T,
        origin='lower',
        cmap='gray_r',
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar(im, ax=ax, label="Proportion of time")

    ax.set_title("Occupancy Map")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            os.path.join(save_dir, filename),
            dpi=300,
            bbox_inches="tight"
        )

    return fig, ax


def plot_units_rate_maps(
    rate_maps,
    fill_na=False,
    n_cols=10,
    save_dir=None,
    filename=None,
    vmin=0,
    vmax=3
):
    """
    Plot rate maps using consistent (y, x) convention.
    """

    n_units = rate_maps.shape[0]
    n_rows = int(np.ceil(n_units / n_cols))

    data = rate_maps.copy()

    if fill_na:
        data = np.where(np.isnan(data), vmin, data)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2, n_rows * 2)
    )

    axes = np.array(axes).reshape(n_rows, n_cols)

    for unit in range(n_units):
        ax = axes[unit // n_cols, unit % n_cols]

        im = ax.imshow(
            data[unit],          # ❗ NO TRANSPOSE
            origin='lower',
            cmap='viridis',
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(f"Unit {unit}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for i in range(n_units, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    # Colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Activation")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            os.path.join(save_dir, filename),
            dpi=300,
            bbox_inches="tight"
        )

    return fig, axes

def plot_average_units_rate_map(rate_maps, fill_na=False, save_dir=None, filename="rate_map_avg_units.png",vmin=0, vmax=3):
    """
    Plot a single rate map averaging activation across all units.

    Args:
        rate_maps: np.array of shape (n_units, n_bins, n_bins)
        fill_na:   whether to fill NaN values with the minimum visited value
        save_dir:  directory to save the plot (optional)
        filename:  filename for saving the plot
    """

    # Average across units -> shape (n_bins, n_bins)
    rate_map = np.nanmean(rate_maps, axis=0)

    # vmin = np.nanmin(rate_map)
    # vmax = np.nanmax(rate_map)
    if fill_na:
        rate_map = np.where(np.isnan(rate_map), vmin, rate_map)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(rate_map.T, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Average activation")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")

    return fig, ax



def plot_multiple_models(
    results_dict,
    x_label,
    y_label,
    save=False,
    filename=None
):
    """
    results_dict:
        {model_name: (mean, ci_low, ci_high)}
    """

    fig, ax = plt.subplots(figsize=(8, 4))

    for model, (mean, low, high) in results_dict.items():

        mean = np.asarray(mean)
        low = np.asarray(low)
        high = np.asarray(high)

        x = np.arange(len(mean))

        ax.plot(x, mean, lw=2, label=model)
        ax.fill_between(x, low, high, alpha=0.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.legend()

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save:
        if filename is None:
            raise ValueError("filename must be provided when save=True")

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    return fig, ax