import ratinabox
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells
from ratinabox.contribs.NeuralNetworkNeurons import NeuralNetworkNeurons #for the Actor and Critic
from ratinabox.contribs.TaskEnvironment import (SpatialGoalEnvironment, SpatialGoal, Reward)

#misc
import torch 
import numpy as np 
from gymnasium.spaces import Box
from tqdm import tqdm
import time
import random

from plotting_utils import *
from base_actor_critic import *
from training_utils import *

#TASK CONSTANTS
DT = 0.1 # Time step
T_TIMEOUT = 0.3 # Time out
GOAL_POS = np.array([0.9, 0.9]) # Goal position
WALL = None
# WALL = [[0.8, 0.0], [0.8, 0.8]]
GOAL_RADIUS = 0.1
REWARD = 1 # Reward
REWARD_DURATION = 1 # Reward duration

#LEARNING CONSTANTS
TAU = 5 # Discount time horizon
TAU_E = 5 # Eligibility trace time horizon
ETA = 0.01 # Learning rate 
N_EPISODES = 5000 # Number of episodes
L2 = 0.000 # L2 regularization


env = SpatialGoalEnvironment(
        dt=DT,
        teleport_on_reset=True, # change this to False if you want the agent to start at the same position at the start of each episode
        episode_terminate_delay=REWARD_DURATION)
env.exploration_strength = 0 
if WALL is not None: 
    env.add_wall(WALL)
#Make the reward which is given when a spatial goal is satisfied. Attached this goal to the environment
reward = Reward(REWARD,decay="none",expire_clock=REWARD_DURATION,dt=DT,)
goals = [SpatialGoal(env,pos=GOAL_POS,goal_radius=GOAL_RADIUS, reward=reward)]
env.goal_cache.reset_goals = goals 
#Recruit the agent and add it to environment
ag = Agent(env,params={'dt':DT})
env.add_agents(ag)

placecells = PlaceCells(ag, params={'n':50,}) 

#Make the actor and the critic (first make their core NNs then pass these to the full Actor and Critic classes)
# actorNN  = VxVyGaussianFTU(n_in=placecells.n)
actorNN  = VxVyGaussianFTA(n_in=placecells.n,post_fta=[2])
criticNN = FTANetwork(n_in=placecells.n,post_fta=[1])
#
default_params_actor = {
        "tau": TAU, #The time horizon of the value function 
        "tau_z": TAU_E, #The time horizon of the eligibility trace
        "input_layers": [placecells],  # a list of input layers, each must be a ratinabox.Neurons class
        "NeuralNetworkModule": actorNN, #Any torch nn.Sequential or nn.Module with a .forward() method
        "optimizer": lambda params: torch.optim.SGD(params, lr=ETA,  maximize=True, weight_decay=L2), #The optimizer to use (in practise I've tried Adam but it aint great). Also, remember this must maximize not minimize. 
        "eligibility_traces": False, #Whether to use eligibility traces or not. If False, then the update is based on the current state only, rather than a decaying trace of past states.
        }
actor  = Actor(ag, params = default_params_actor); actor.colormap="PiYG"

default_params_critic = {
        "tau": TAU, #The time horizon of the value function 
        "tau_z": TAU_E, #The time horizon of the eligibility trace
        "input_layers": [placecells],  # a list of input layers, each must be a ratinabox.Neurons class
        "NeuralNetworkModule": criticNN, #Any torch nn.Sequential or nn.Module with a .forward() method
        "optimizer": lambda params: torch.optim.SGD(params, lr=ETA,  maximize=True, weight_decay=L2), #The optimizer to use (in practise I've tried Adam but it aint great). Also, remember this must maximize not minimize. 
        "eligibility_traces": False, #Whether to use eligibility traces or not. If False, then the update is based on the current state only, rather than a decaying trace of past states.
        }
critic = Critic(ag, params = default_params_critic)

train_agent_episodes(env, ag, actor, critic, placecells, n_episodes=N_EPISODES, time_limit=T_TIMEOUT)

#This plots the reward history over training
fig, ax = plot_reward_history(env)
#This visualises the value function and "policy" over the entire environment
fig, ax = critic.plot_rate_map(); fig.suptitle("Value function (before learning)")
fig, ax = actor.plot_rate_map(zero_center=True); fig.suptitle("Policy (before learning)"); ax[0].set_title("Vx"); ax[1].set_title("Vy")
#Shows the trajectory of the agent over the last 100 episodes
fig, ax = ag.plot_trajectory(color="changing",t_start=env.episodes['start'][0],t_end=env.episodes['start'][10]); display_reward_patch(fig,ax, reward_pos=GOAL_POS, reward_radius=0.1)