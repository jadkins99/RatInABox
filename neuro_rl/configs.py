import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
DATA_DIR = os.path.join(BASE_DIR, "data")

N_BINS = 20
N_PLACECELLS = 5

#TASK CONSTANTS
DT = 0.1 # Time step
T_TIMEOUT = 15 # Time out
GOAL_POS = np.array([0.5, 0.5]) # Goal position
WALL = None
# WALL = [[0.8, 0.0], [0.8, 0.8]]
GOAL_RADIUS = 0.1
REWARD = 1 # Reward
REWARD_DURATION = 1 # Reward duration
OBSTACLES = {
    "empty": [],
    "obstacle_near_goal": [{"x_min": 0.30, "x_max": 0.45, "y_min": 0.30, "y_max": 0.45}],
    "obstacle_far_goal":  [{"x_min": 0.85, "x_max": 1.0,  "y_min": 0.85, "y_max": 1.0}],
}

#LEARNING CONSTANTS
FTA_ETA = 0.1 # FTA learning rate
TAU = 5 # Discount time horizon
TAU_E = 5 # Eligibility trace time horizon
ETA = 0.01 # Learning rate 
N_EPISODES = 3 # Number of episodes
L2 = 0.000 # L2 regularization

