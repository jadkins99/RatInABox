import os
import random
import numpy as np
import torch
import tqdm
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells 
from ratinabox.contribs.TaskEnvironment import (SpatialGoalEnvironment, SpatialGoal, Reward)

from .base_actor_critic import Actor, Critic, VxVyGaussianMLP, FTANetwork
from .training_utils import run_episode
from .hook import create_fta_hook
from .configs import *


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
    t_timeout=15,
    input_min=-1.0,
    input_max=1.0,
    goal_pos=np.array([0.9, 0.9]),
    goal_radius=0.1,
    reward_val=1,
    reward_duration=1,
    wall=None,
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
    env = SpatialGoalEnvironment(dt=dt, teleport_on_reset=True, episode_terminate_delay=reward_duration)
    env.exploration_strength = 0
    if wall is not None:
        env.add_wall(wall)
    
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
    criticNN = FTANetwork(n_in=placecells.n, input_min=input_min, input_max=input_max, post_fta=critic_fta_post, eta=fta_eta)
    
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
            # ag.reset()  # Uncomment if your agent has a reset method

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
            if success_frac > 0.99 and i > 10: break
    except KeyboardInterrupt:
        print("Interrupted by user")

    
    return all_episode_sparsity, all_episode_time, all_episodes_state,all_episode_bins, all_episode_bins_sparsity, all_out_arrays


def run_experiment(num_runs):

    runs_out_arrays = []
    runs_timesteps = []
    runs_bins = []
    runs_states = []

    for run in range(num_runs):

        set_seed(run)
        env, ag, placecells, actor, critic = create_experiment()
        fta = critic.ftanetwork.fta
        all_episode_sparsity, all_episode_time, all_episodes_state, all_episode_bins, all_episode_bins_sparsity, all_out_arrays = run_multiple_episodes_with_fta(
        n_episodes=N_EPISODES,
        env=env,
        ag=ag,
        actor=actor,
        critic=critic,
        placecells=placecells,
        fta=fta,
        num_bins=N_BINS,
        time_limit=T_TIMEOUT
        )

        runs_out_arrays.append(all_out_arrays)
        runs_timesteps.append(all_episode_time)
        runs_bins.append(all_episode_bins)
        runs_states.append(all_episodes_state)