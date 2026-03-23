import os
import random
import numpy as np
import torch
from tqdm import tqdm
import argparse

from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells 
from ratinabox.contribs.TaskEnvironment import (SpatialGoalEnvironment, SpatialGoal, Reward)

from base_actor_critic import Actor, Critic, VxVyGaussianMLP, FTANetwork
from utils import *
from training_utils import run_episode
from hook import create_fta_hook, find_layer_module
from configs import *
from plotting_utils import plot_bin_counts_per_percentage, plot_dead_neurons_over_time, plot_bin_counts_per_percentage, plot_fta_average_units_rate_map, plot_occupancy_map, plot_rate_maps, plot_fta_rate_maps
from analysis import compute_dead_neurons_per_timestep, compute_dead_neurons_per_episode,compute_bin_counts_per_timestep, compute_fta_rate_map_single

def get_environment(shape="empty", dt=DT, episode_terminate_delay=REWARD_DURATION, teleport_on_reset=True):
    if shape == "empty":
        env = SpatialGoalEnvironment(dt=dt, teleport_on_reset=teleport_on_reset, episode_terminate_delay=episode_terminate_delay)
    elif shape == "obstacle_near_goal":
        env = SpatialGoalEnvironment(dt=dt, teleport_on_reset=teleport_on_reset, episode_terminate_delay=episode_terminate_delay)
        env.add_wall([[.30, .30], [.30, .45]])
        env.add_wall([[.30, .45], [.45, .45]])
        env.add_wall([[.45, .45], [.45, .30]])
        env.add_wall([[.45, .30], [.30, .30]])
        resolution = 0.01  # spacing between walls
        for y in np.arange(0.30, 0.45, resolution):
            env.add_wall([[.30, y], [.45, y]])
    elif shape == "obstacle_far_goal":
        env = SpatialGoalEnvironment(dt=dt, teleport_on_reset=teleport_on_reset, episode_terminate_delay=episode_terminate_delay)
        env.add_wall([[0.85, 0.85], [0.85, 1.0]])
        env.add_wall([[0.85, 1.0], [1.0, 1.0]])
        env.add_wall([[1.0, 1.0], [1.0, 0.85]])
        env.add_wall([[1.0, 0.85], [0.85, 0.85]])
        resolution = 0.01
        for y in np.arange(0.85, 1.0, resolution):
            env.add_wall([[0.85, y], [1.0, y]])
    else:
        raise ValueError(f"Unknown environment shape: {shape}")
    return env


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

def create_experiment(
    dt=0.1,
    env_shape="empty",
    t_timeout=15,
    input_min=-1.0,
    input_max=1.0,
    n_tiles=N_BINS,
    goal_pos=np.array([0.9, 0.9]),
    goal_radius=0.1,
    reward_val=1,
    reward_duration=1,
    n_placecells=50,
    fta_eta=0.1,
    eta = 0.01,
    l2=0.0,
    tau=5,
    tau_e=5,
    actor_fta_post=[2],
    critic_fta_post=[1],
):
    """Returns a fresh environment, agent, place cells, actor, critic"""
    
    # --- Environment ---
    env = get_environment(shape=env_shape, dt=dt, episode_terminate_delay=reward_duration)
    env.exploration_strength = 0
    
    # Reward and goal
    reward = Reward(reward_val, decay="none", expire_clock=reward_duration, dt=dt)
    goals = [SpatialGoal(env, pos=goal_pos, goal_radius=goal_radius, reward=reward)]
    env.goal_cache.reset_goals = goals
    
    # --- Agent ---
    ag = Agent(env, params={'dt': dt})
    env.add_agents(ag)
    
    # --- Place Cells ---
    placecells = PlaceCells(ag, params={'n': n_placecells})
    
    # --- Networks ---
    # actorNN = VxVyGaussianFTA(n_in=placecells.n, post_fta=actor_fta_post)
    actorNN = VxVyGaussianMLP(n_in=placecells.n) #add seed
    criticNN = FTANetwork(n_in=placecells.n, n_out=1,n_tiles=n_tiles, input_min=input_min, input_max=input_max, post_fta=critic_fta_post, eta=fta_eta)
    
    # --- Actor ---
    default_params_actor = {
        "tau": tau,
        "tau_z": tau_e,
        "input_layers": [placecells],
        "NeuralNetworkModule": actorNN,
        "optimizer": lambda params: torch.optim.SGD(params, lr=eta, maximize=True, weight_decay=l2),
        "eligibility_traces": True,
    }
    actor = Actor(ag, params=default_params_actor)
    actor.colormap = "PiYG"
    
    # --- Critic ---
    default_params_critic = {
        "tau": tau,
        "tau_z": tau_e,
        "input_layers": [placecells],
        "NeuralNetworkModule": criticNN,
        "optimizer": lambda params: torch.optim.SGD(params, lr=eta, maximize=True, weight_decay=l2),
        "eligibility_traces": True,
    }
    critic = Critic(ag, params=default_params_critic)
    
    return env, ag, placecells, actor, critic


def run_multiple_episodes_with_fta(
    n_episodes,
    env,
    ag,
    actor,
    critic,
    placecells,
    fta,
    num_bins,
    time_limit=15
):
    """
    Runs multiple episodes, records FTA sparsity at each timestep.

    Returns:
        all_episode_sparsity: list of lists of sparsity values (one per episode)
        all_episode_time: list of lists of time steps (one per episode)
    """
    
    all_episode_sparsity = []
    all_episode_time = []
    all_episodes_state = []
    all_episode_bins_sparsity = []
    all_episode_bins = []
    all_out_arrays = []
    

    try: 
        for i in (pbar := tqdm(range(n_episodes), desc = "")):
            # Reset environment and agent
            env.reset()

            # Create new FTA hook for this episode
            hook_fn, fta_sparsity, fta_states, fta_times, fta_bins, fta_bins_sparsity, out_arrays = create_fta_hook(fta, ag, env, bins=num_bins)
            handle = fta.register_forward_hook(hook_fn)

            
            # Run one episode
            run_episode(env, ag, actor, critic, state_cells=[placecells], time_limit=time_limit, seed=i)

            # Remove hook
            handle.remove()

            # Store episode data
            all_episode_sparsity.append(fta_sparsity)
            all_episode_time.append(fta_times)
            all_episodes_state.append(fta_states)
            all_episode_bins_sparsity.append(fta_bins_sparsity)
            all_episode_bins.append(fta_bins)
            all_out_arrays.append(out_arrays)
            
            
            success_frac = np.mean(np.array(env.episodes['meta_info'][-100:]) == "completed")
            episode_time = np.mean(env.episodes['duration'][-100:])
            pbar.set_description(f"<success fraction>: {success_frac:.2f}, <episode time> {episode_time:.1f}")
            if success_frac > 0.99 and i > 10:
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    
    return all_episode_sparsity, all_episode_time, all_episodes_state,all_episode_bins, all_episode_bins_sparsity, all_out_arrays


def run_experiment(seed, env_shape="empty"):

    print(f"Starting experiment (seed {seed}") 

    set_seed(seed)
    env, ag, placecells, actor, critic = create_experiment(dt=DT, env_shape=env_shape, t_timeout=T_TIMEOUT, input_min=-1, input_max=1, goal_pos=GOAL_POS, goal_radius=GOAL_RADIUS, reward_val=REWARD, reward_duration=REWARD_DURATION, n_placecells=N_PLACECELLS, fta_eta=FTA_ETA, eta=ETA, l2=L2, tau=TAU, tau_e=TAU_E)
    fta = find_layer_module(critic.NeuralNetworkModule, 'FTA')
    _, all_episodes_time, all_episodes_state, all_episodes_bins, _, all_out_arrays = run_multiple_episodes_with_fta(n_episodes=N_EPISODES,env=env,ag=ag,actor=actor,critic=critic,placecells=placecells,fta=fta,num_bins=N_BINS,time_limit=T_TIMEOUT)

    save_data(all_episodes_time, os.path.join(DATA_DIR, f'all_episodes_time_seed_{seed}'))
    save_data(all_episodes_state, os.path.join(DATA_DIR, f'all_episodes_states_seed_{seed}'))
    save_data(all_episodes_bins, os.path.join(DATA_DIR, f'all_episodes_bins_seed_{seed}'))
    save_data(all_out_arrays, os.path.join(DATA_DIR, f'all_out_arrays_seed_{seed}'))

    plot_rate_maps(env, ag, placecells, actor, critic, GOAL_POS, GOAL_RADIUS, reward=True, trajectory=True, save_dir=os.path.join(FIGURES_DIR,f"env_{env_shape}", f"seed_{seed}"))

    rate_maps, occupancy = compute_fta_rate_map_single(all_episodes_state, all_out_arrays, n_bins=N_BINS, filter_size=1.5)
    plot_fta_rate_maps(rate_maps, n_cols=N_BINS, save_dir=os.path.join(FIGURES_DIR, f"env_{env_shape}", f"seed_{seed}"), filename=f"fta_rate_maps_seed_{seed}.png")
    plot_occupancy_map(occupancy, save_dir=os.path.join(FIGURES_DIR, f"env_{env_shape}", f"seed_{seed}"), filename=f"occupancy_seed_{seed}.png")

    rate_maps_obs, occupancy_obs = compute_fta_rate_map_single(all_episodes_state, all_out_arrays, n_bins=N_BINS, filter_size=1.5, obstacles=OBSTACLES[env_shape])
    plot_fta_rate_maps(rate_maps_obs, n_cols=N_BINS, save_dir=os.path.join(FIGURES_DIR, f"env_{env_shape}", f"seed_{seed}"), filename=f"fta_rate_maps_obs_seed_{seed}.png")
    plot_occupancy_map(occupancy_obs, save_dir=os.path.join(FIGURES_DIR, f"env_{env_shape}", f"seed_{seed}"), filename=f"occupancy_obs_seed_{seed}.png")

    plot_fta_average_units_rate_map(rate_maps, save_dir=os.path.join(FIGURES_DIR, f"env_{env_shape}", f"seed_{seed}"), filename=f"fta_rate_map_avg_units_seed_{seed}.png")
    plot_fta_average_units_rate_map(rate_maps_obs, save_dir=os.path.join(FIGURES_DIR, f"env_{env_shape}", f"seed_{seed}"), filename=f"fta_rate_map_avg_units_obs_seed_{seed}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_shape", type=str, default="empty")
    args = parser.parse_args()

    run_experiment(seed=args.seed, env_shape=args.env_shape)
    
# First, compute rate maps per seed and then average them.


# envs = ["empty", "obstacle_near_goal", "obstacle_far_goal"]

# for env_shape in envs:

#     # Run experiment and collect FTA outputs
#     runs_out_arrays, runs_states, runs_bins = run_experiment(num_runs=NUM_RUNS, env_shape=env_shape)

#     # Dead neuron analysis
#     dead_neurons_per_timestep = compute_dead_neurons_per_timestep(runs_out_arrays, thres=0.1)
#     plot_dead_neurons_over_time(x=np.arange(len(dead_neurons_per_timestep)), y =dead_neurons_per_timestep, x_label='timesteps', y_label='%\ dead neurons', save=True, filename=os.path.join(FIGURES_DIR, f"env_{env_shape}", "dead_neurons_per_timestep.png"))
#     dead_neurons_per_episode = compute_dead_neurons_per_episode(runs_out_arrays, thres=0.1)
#     plot_dead_neurons_over_time(x=np.arange(len(dead_neurons_per_episode)), y =dead_neurons_per_episode, x_label='episodes', y_label='%\ dead neurons', save=True, filename=os.path.join(FIGURES_DIR, f"env_{env_shape}", "dead_neurons_per_episode.png"))
#     bin_counts = compute_bin_counts_per_timestep(runs_out_arrays, num_bins=N_BINS, threshold=0.1)
#     plot_bin_counts_per_percentage(bin_counts, percentages=[1,2,5,7,10,30,50,70,90,100], save=True, filename=os.path.join(FIGURES_DIR, f"env_{env_shape}"))

#     # Compute and plot FTA rate maps
#     rate_maps_per_run, occupancy_per_run, rate_maps_avg, occupancy_map, average_events = compute_fta_rate_maps(runs_states, runs_out_arrays, n_bins=N_BINS, filter_size=1.5)

#     # Plot average per unit across runs
#     plot_fta_rate_maps(rate_maps_avg, n_cols=N_BINS, save_dir=os.path.join(FIGURES_DIR, f"env_{env_shape}"), filename="fta_rate_maps_avg.png")
#     plot_occupancy_map(occupancy_map, save_dir=os.path.join(FIGURES_DIR, f"env_{env_shape}"), filename="occupancy_avg.png")

#     # Plot per run
#     for run_idx, (rate_maps, occupancy) in enumerate(zip(rate_maps_per_run, occupancy_per_run)):
#         plot_fta_rate_maps(rate_maps, n_cols=N_BINS, save_dir=os.path.join(FIGURES_DIR, f"env_{env_shape}", f"run_{run_idx+1}"), filename=f"fta_rate_maps_run{run_idx}.png")
#         plot_occupancy_map(occupancy, save_dir=os.path.join(FIGURES_DIR, f"env_{env_shape}", f"run_{run_idx+1}"), filename=f"occupancy_run{run_idx}.png")

#     # Average across runs
#     plot_fta_average_units_rate_map(rate_maps_avg, save_dir=os.path.join(FIGURES_DIR, f"env_{env_shape}"), filename=f"fta_rate_map_avg_units.png")

#     # Per run
#     for run_idx, rate_maps in enumerate(rate_maps_per_run):
#         plot_fta_average_units_rate_map(rate_maps, save_dir=os.path.join(FIGURES_DIR, f"env_{env_shape}", f"run_{run_idx+1}"), filename=f"fta_rate_map_avg_units_run{run_idx}.png")

 
    