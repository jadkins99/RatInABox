import numpy as np
import random
from tqdm import tqdm
from ratinabox.Agent import Agent
from ratinabox.contribs.TaskEnvironment import (SpatialGoalEnvironment, SpatialGoal, Reward)
from plotting_utils import *

def run_episode(env : SpatialGoalEnvironment, 
                ag, 
                actor, 
                critic,
                seed,
                state_cells = [],
                time_limit=15,
                egocentric_actions = False):
    
    """Run an episode of the agent in the environment.
    Returns 1 if the episode timed out, 0 otherwise.
    """
    # reset the actor and the critic
    critic.initialise_traces()
    actor.initialise_traces()

    while True:
        # SAMPLE ACTION AND ITS LOG PROB
        action, log_prob = actor.NeuralNetworkModule.sample_action(actor.firingrate_torch)
        if egocentric_actions: 
            action = ego_to_allo(action, ag.head_direction) #convert action in in [V_leftright, V_forwardbackward] (ego) to [V_x, V_y] (allo)
        
        # STEP THE ENVIRONMENT AND OBSERVE THE REWARD
        _, reward, terminate_episode, _ , _ =  env.step1(action=action,
                                                         drift_to_random_strength_ratio=1,)
        # wall_penalty = WALL_PENALTY * (ag.distance_to_closest_wall < 0.1)

        # UPDATE THE STATE CELLS
        for cell in state_cells:
            cell.update()

        # UPDATE THE CRITIC AND ACTOR (INCLUDING LEARNING)
        critic.update(reward=reward)
        actor.update(log_prob=log_prob, td_error=critic.td_error)

        # CHECK IF THE EPISODE IS OVER
        if env.t - env.episodes['start'][-1] > time_limit: 
            env.episodes['meta_info'].append("timeout")
            return
        elif terminate_episode:
            env.episodes['meta_info'].append("completed")
            return
        
def train_agent_episodes(
    env,
    ag,
    actor,
    critic,
    placecells,
    n_episodes,
    success_window=100,
    success_threshold=0.99,
    min_episodes_before_stop=10,
    seed = 0,
    time_limit = 15,
):
    """
    Runs training episodes and stops early if performance threshold is reached.

    """

    try:
        # for i in (pbar := tqdm(range(n_episodes))):
        for i in range(n_episodes):
    
            run_episode(
                env,
                ag,
                actor,
                critic,
                state_cells=[placecells],
                time_limit= time_limit,
                seed = i
            )

            success_frac = np.mean(
                np.array(env.episodes["meta_info"][-success_window:]) == "completed"
            )

            episode_time = np.mean(
                env.episodes["duration"][-success_window:]
            )

            # pbar.set_description(
            #     f"<success fraction>: {success_frac:.2f}, "
            #     f"<episode time>: {episode_time:.1f}"
            # )

            if success_frac > success_threshold and i > min_episodes_before_stop:
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

def sparsity_function(fta_arr: np.ndarray, thres=0.01):
    """Compute sparsity of a FTA activation array"""
    sparsity_num = 1 - np.sum(fta_arr[fta_arr > thres]) / fta_arr.size
    return sparsity_num


def create_fta_hook(fta_module, agent, env):
    """Returns a hook function that collects sparsity"""
    fta_sparsity = []
    fta_states = []
    fta_times = []

    def hook(module, inputs, output):
        fta_arr = output.detach().cpu().numpy().flatten()
        sparsity = sparsity_function(fta_arr)
        fta_sparsity.append(sparsity)

        # Optional: store the agent position at this step
        fta_states.append(np.copy(agent.pos))
        fta_times.append(env.t)
    return hook, fta_sparsity, fta_states, fta_times