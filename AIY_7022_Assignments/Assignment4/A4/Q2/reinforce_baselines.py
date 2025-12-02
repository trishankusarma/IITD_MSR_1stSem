import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    """Gaussian policy network for continuous actions"""
    
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, x):
        x = self.layers(x)
        mean = self.mean(x)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        return mean, std


class ValueNetwork(nn.Module):
    """Value network for value function baseline"""
    
    def __init__(self, input_dim, hidden_dim=128): 
        super(ValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.layers(x).squeeze(-1)

def compute_returns_no_baseline(batch_rewards, gamma):
    """
    For NO BASELINE: Compute total episode return G for each timestep.
    All timesteps in an episode get the SAME return value.
    
    G = sum_{t=0}^{T-1} gamma^t * r_t
    """
    all_returns = []
    for ep_rewards in batch_rewards:
        # Compute total discounted return for this episode
        G = sum((gamma**t) * r for t, r in enumerate(ep_rewards))
        # All timesteps in episode get same return
        episode_length = len(ep_rewards)
        all_returns.extend([G] * episode_length)
    
    all_returns = np.array(all_returns, dtype=np.float32)
    # Normalize
    all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-8)
    return all_returns

def compute_returns_reward_to_go(batch_rewards, gamma):
    """
    For REWARD-TO-GO: Compute reward-to-go G_t for each timestep.
    Different timesteps get different values.
    
    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    """
    all_returns = []
    for ep_rewards in batch_rewards:
        returns_episode = []
        G = 0.0
        # Work backwards from end of episode
        for r in reversed(ep_rewards):
            G = r + gamma * G
            returns_episode.insert(0, G)
        all_returns.extend(returns_episode)
    
    all_returns = np.array(all_returns, dtype=np.float32)
    # Normalize
    all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-8)
    return all_returns

def compute_returns_avg_baseline(batch_rewards, gamma):
    """
    For AVERAGE REWARD BASELINE: 
    1. Compute total returns for all episodes
    2. Subtract mean of all returns as baseline
    """
    all_returns = []
    for ep_rewards in batch_rewards:
        # Compute total discounted return
        G = sum((gamma**t) * r for t, r in enumerate(ep_rewards))
        episode_length = len(ep_rewards)
        all_returns.extend([G] * episode_length)
    
    all_returns = np.array(all_returns, dtype=np.float32)
    # Baseline is mean of all returns in this batch
    baseline = np.mean(all_returns)
    # Return advantages (returns - baseline)
    return all_returns - baseline

def compute_advantages_value_baseline(batch_rewards, batch_states, value_net, gamma, device):
    """
    For VALUE FUNCTION BASELINE:
    1. Compute reward-to-go returns
    2. Get value estimates from network
    3. Advantages = returns - values
    4. Normalize advantages
    """
    # Compute reward-to-go returns
    all_returns = []
    for ep_rewards in batch_rewards:
        returns_episode = []
        G = 0.0
        for r in reversed(ep_rewards):
            G = r + gamma * G
            returns_episode.insert(0, G)
        all_returns.extend(returns_episode)
    
    returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=device)
    
    # Get value estimates
    states_tensor = torch.tensor(np.array(batch_states, dtype=np.float32), 
                                  dtype=torch.float32, device=device)
    with torch.no_grad():
        values = value_net(states_tensor)
    
    # Compute advantages
    advantages = returns_tensor - values
    
    # Normalize advantages
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    normalized_advantages = (advantages - adv_mean) / adv_std
    
    return normalized_advantages.cpu().numpy(), returns_tensor.cpu().numpy()