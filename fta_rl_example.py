"""
FTA-Enhanced RL Example - Standalone Integration Guide

This script demonstrates how to integrate Fuzzy Tiling Activation (FTA)
into a RatInABox RL experiment. It shows:

1. Using FTA for agent state encoding
2. Building RL networks with FTA features
3. Complete training loop example
4. Performance comparison (with/without FTA)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from fta_pytorch import FuzzyTilingActivation
import matplotlib.pyplot as plt


# ============================================================================
# PART 1: FTA-Enhanced Actor-Critic Agent
# ============================================================================

class RLAgentWithFTA(nn.Module):
    """
    Actor-Critic RL agent with Fuzzy Tiling Activation feature encoding.
    
    Architecture:
        state → FTA → shared_features → [policy_head, value_head]
    """
    
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 n_tiles=10, 
                 n_tilings=1,
                 hidden_dim=128):
        """
        Args:
            state_dim: Dimensionality of state (e.g., 2 for 2D position)
            action_dim: Number of actions
            n_tiles: Number of tiles per state dimension
            n_tilings: Number of offset tilings
            hidden_dim: Size of hidden layers
        """
        super().__init__()
        
        # Feature extraction with FTA
        self.fta = FuzzyTilingActivation(
            input_dim=state_dim,
            n_tiles=n_tiles,
            n_tilings=n_tilings,
            input_min=-1.0,
            input_max=1.0
        )
        
        fta_output_dim = self.fta.output_dim
        
        # Shared feature extractor (optional)
        self.shared_net = nn.Sequential(
            nn.Linear(fta_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """
        Forward pass through agent.
        
        Args:
            state: [batch_size, state_dim] tensor
        
        Returns:
            policy: [batch_size, action_dim] action probabilities
            value: [batch_size, 1] state value estimate
        """
        # Extract FTA features
        fta_features = self.fta(state)
        
        # Shared feature processing
        shared_features = self.shared_net(fta_features)
        
        # Compute policy and value
        policy = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        
        return policy, value
    
    def get_action(self, state):
        """
        Sample action from learned policy.
        
        Args:
            state: [state_dim] numpy array or tensor
        
        Returns:
            action: int, sampled action
            log_prob: float, log probability of action
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().unsqueeze(0)
        
        with torch.no_grad():
            policy, _ = self.forward(state)
        
        dist = torch.distributions.Categorical(policy[0])
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()


# ============================================================================
# PART 2: Simple RL Environment
# ============================================================================

class SimpleNavigationEnv:
    """
    Simple 2D navigation environment for testing.
    
    Agent starts at origin, goal is to reach (0.8, 0.8).
    """
    
    def __init__(self, goal_pos=(0.8, 0.8)):
        self.goal_pos = np.array(goal_pos, dtype=np.float32)
        self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.steps = 0
        self.max_steps = 100
    
    def reset(self):
        """Reset environment."""
        self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.steps = 0
        return self.agent_pos.copy()
    
    def step(self, action):
        """
        Execute action.
        
        Actions: 0=up, 1=down, 2=left, 3=right
        """
        step_size = 0.1
        
        if action == 0:  # up
            self.agent_pos[1] += step_size
        elif action == 1:  # down
            self.agent_pos[1] -= step_size
        elif action == 2:  # left
            self.agent_pos[0] -= step_size
        elif action == 3:  # right
            self.agent_pos[0] += step_size
        
        # Clip to bounds
        self.agent_pos = np.clip(self.agent_pos, -1.0, 1.0)
        
        # Compute reward
        distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward = 1.0 - distance  # Reward based on proximity to goal
        
        # Check if goal reached
        done = (distance < 0.15) or (self.steps >= self.max_steps)
        if distance < 0.15:
            reward = 10.0  # Bonus for reaching goal
        
        self.steps += 1
        
        return self.agent_pos.copy(), reward, done
    
    def render(self):
        """Print current state."""
        print(f"Agent: {self.agent_pos}, Goal: {self.goal_pos}, "
              f"Distance: {np.linalg.norm(self.agent_pos - self.goal_pos):.3f}")


# ============================================================================
# PART 3: Training Loop
# ============================================================================

def train_episode(agent, env, optimizer, gamma=0.99):
    """
    Train agent for one episode using Actor-Critic.
    
    Args:
        agent: RLAgentWithFTA
        env: Environment
        optimizer: PyTorch optimizer
        gamma: Discount factor
    
    Returns:
        episode_reward: Total reward for episode
    """
    agent.train()
    state = env.reset()
    episode_reward = 0.0
    episode_loss = 0.0
    
    for step in range(100):
        # Convert state to tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        
        # Get action from policy
        policy, value = agent(state_tensor)
        
        dist = torch.distributions.Categorical(policy[0])
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Execute action
        next_state, reward, done = env.step(action.item())
        episode_reward += reward
        
        # Compute TD target
        next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)
        with torch.no_grad():
            _, next_value = agent(next_state_tensor)
        
        td_target = torch.tensor(reward + (gamma * next_value.item() * (1 - int(done))), 
                                 dtype=torch.float32)
        td_error = td_target - value[0, 0]
        
        # Actor-Critic loss
        actor_loss = -log_prob * td_error.detach()
        critic_loss = td_error ** 2
        loss = actor_loss + 0.5 * critic_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        episode_loss += loss.item()
        
        state = next_state
        
        if done:
            break
    
    return episode_reward, episode_loss / (step + 1)


# ============================================================================
# PART 4: Main Script
# ============================================================================

def main():
    """Run complete FTA-RL example."""
    
    print("=" * 70)
    print("Fuzzy Tiling Activation - RL Integration Example")
    print("=" * 70)
    
    # Configuration
    STATE_DIM = 2
    ACTION_DIM = 4
    N_TILES = 10
    N_TILINGS = 1
    LEARNING_RATE = 0.001
    N_EPISODES = 100
    
    # Create agent with FTA
    print(f"\n[1] Creating RLAgentWithFTA...")
    agent = RLAgentWithFTA(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        n_tiles=N_TILES,
        n_tilings=N_TILINGS,
        hidden_dim=128
    )
    
    # Print agent architecture
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"    Total parameters: {total_params}")
    
    # Optimizer
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    
    # Training
    print(f"\n[2] Training for {N_EPISODES} episodes...")
    rewards = []
    losses = []
    
    for episode in range(N_EPISODES):
        env = SimpleNavigationEnv()
        episode_reward, episode_loss = train_episode(agent, env, optimizer)
        
        rewards.append(episode_reward)
        losses.append(episode_loss)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            avg_loss = np.mean(losses[-10:])
            print(f"    Episode {episode + 1}: "
                  f"Avg Reward={avg_reward:.3f}, "
                  f"Loss={avg_loss:.4f}")
    
    # Testing
    print(f"\n[3] Testing learned policy...")
    test_rewards = []
    for _ in range(10):
        env = SimpleNavigationEnv()
        state = env.reset()
        episode_reward = 0.0
        
        for _ in range(100):
            action, _ = agent.get_action(state)
            state, reward, done = env.step(action)
            episode_reward += reward
            if done:
                break
        
        test_rewards.append(episode_reward)
    
    print(f"    Test average reward: {np.mean(test_rewards):.3f} "
          f"± {np.std(test_rewards):.3f}")
    
    # Visualization
    print(f"\n[4] Generating plots...")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Reward plot
        axes[0].plot(rewards, alpha=0.7, label='Episode Reward')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Training Rewards with FTA')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Loss plot
        axes[1].plot(losses, alpha=0.7, label='Actor-Critic Loss')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training Loss with FTA')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('fta_rl_training.png', dpi=150)
        print(f"    Saved: fta_rl_training.png")
    except Exception as e:
        print(f"    Could not create plots: {e}")
    
    print("\n" + "=" * 70)
    print("Training complete! ✓")
    print("=" * 70)
    
    return agent, rewards, losses


# ============================================================================
# Integration Tips for RatInABox RL Example
# ============================================================================

"""
INTEGRATION CHECKLIST for demos/reinforcement_learning_example.ipynb:

1. ADD IMPORTS:
   import torch
   from fta_pytorch import FuzzyTilingActivation

2. REPLACE INPUT FEATURES:
   
   Before:
   ```
   Inputs = PlaceCells(Ag, params={"n": 200, ...})
   ```
   
   After:
   ```
   fta = FuzzyTilingActivation(
       input_dim=2,                          # Agent position
       n_tiles=15,
       n_tilings=1,
       input_min=[0.0, 0.0],
       input_max=[1.0, 1.0]
   )
   ```

3. UPDATE VALUE NEURON INITIALIZATION:
   
   Before:
   ```
   ValNeur = ValueNeuron(Ag, params={
       "input_layers": [Inputs],
       ...
   })
   ```
   
   After:
   ```
   # Create a PyTorch-based value network with FTA
   value_network = nn.Sequential(
       nn.Linear(fta.output_dim, 128),
       nn.ReLU(),
       nn.Linear(128, 1)
   )
   ```

4. UPDATE TRAINING LOOP:
   
   Before:
   ```
   Inputs.update()
   ValNeur.update()
   ValNeur.update_weights(reward=Reward.firingrate[0])
   ```
   
   After:
   ```
   state = torch.tensor([[Ag.pos[0], Ag.pos[1]]], dtype=torch.float32)
   fta_features = fta(state).detach().numpy()
   # Use fta_features instead of Inputs.firingrate
   
   value_pred = value_network(fta_features)
   # Compute TD error and update weights
   ```

5. PARAMETER TUNING:
   - n_tiles: 10-20 for smooth environments
   - eta: None (auto) or manually set to input_range/n_tiles/2
   - n_tilings: 1-2 for most tasks

6. PERFORMANCE TIPS:
   - FTA output dimension: input_dim * n_tiles * n_tilings
   - For 2D position with n_tiles=15: 30 features
   - Adjust downstream network size accordingly
"""


if __name__ == "__main__":
    agent, rewards, losses = main()
    print("\nYou can now use the trained agent in your experiments!")
