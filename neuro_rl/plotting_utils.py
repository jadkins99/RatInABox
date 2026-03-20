import os
import matplotlib 
import matplotlib.pyplot as plt 
import numpy as np
import ratinabox

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

def plot_rate_maps(env, ag, actor, critic, goal_pos, reward_radius, reward=False, trajectory=False, save_dir=None):
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if reward:
        fig, ax = plot_reward_history(env)
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "reward_history.png"))

    fig, ax = critic.plot_rate_map()
    fig.suptitle("Value function (before learning)")
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "value_function.png"))

    fig, ax = actor.plot_rate_map(zero_center=True)
    fig.suptitle("Policy (before learning)")
    ax[0].set_title("Vx")
    ax[1].set_title("Vy")
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "policy.png"))

    if trajectory:
        fig, ax = ag.plot_trajectory(
            color="changing",
            t_start=env.episodes['start'][0],
            t_end=env.episodes['start'][0]
        )
        display_reward_patch(fig, ax, reward_pos=goal_pos, reward_radius=reward_radius)
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "trajectory.png"))

def plot_dead_neurons_over_time(x,y,x_label,y_label, save = False, filename = None):

    plt.figure(figsize=(8,4))
    plt.plot(x, y, color='red', lw=2)
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
        plt.xlabel("FTA Bin")
        plt.ylabel("Active neuron count")
        plt.title(f"Active neurons per bin at {perc}% of training (timestep {idx})")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(False)
        if save:
            plt.savefig(f"{filename}/bin_counts_{perc}percent.png", dpi=300, bbox_inches="tight")
       