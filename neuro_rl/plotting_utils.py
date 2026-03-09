#visualisations
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