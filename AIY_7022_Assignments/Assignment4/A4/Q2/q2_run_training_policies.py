# Code for Q1
import os
import sys
import random
from datetime import datetime
from collections import deque
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm
from utils.utils import plot_training_curve
from reinforce_baselines import (
    PolicyNetwork,
    ValueNetwork,
    compute_returns_no_baseline,
    compute_returns_reward_to_go,
    compute_returns_avg_baseline,
    compute_advantages_value_baseline
)
from utils.loggerUtils import Logger

# HYPERPARAMETERS
SEED = 42
ENV_NAME = "InvertedPendulum-v4"

# Training parameters
GAMMA = 0.99
BATCH_SIZE = 5  # Number of episodes to collect before update
TARGET_REWARD = 500
NUM_EPISODES = 2000
MAX_episode_LENGTH = 1000

# Network parameters
POLICY_LR = 2e-3
VALUE_LR = 2e-3
HIDDEN_DIM = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def logging(s='1'):
    global current_logger
    log_path = os.path.join('logs', f'output_{s}.txt')

    # If stdout is already a Logger, unwrap to real stdout
    if isinstance(sys.stdout, Logger):
        sys.stdout = sys.stdout.terminal

    # Initialize new logger (always starts fresh)
    current_logger = Logger(log_path)
    sys.stdout = current_logger

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"Logging started at {timestamp}")
    print(f"Log file   : {log_path}")
    print(f"Using device: {device}")
    print("=" * 70)

# Reproducibility
def set_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    if env is not None:
        try:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        except Exception:
            pass

# EPISODE GENERATION
def generate_episode(env, policy, action_low, action_high):
    """
    Generate one episode using the current policy.
    """
    states, actions, rewards, log_probs = [], [], [], []
    
    state, _ = env.reset()
    done = False
    
    for _ in range(MAX_episode_LENGTH):
        # Convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Get action from policy
        mean, std = policy(state_tensor)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        action_np = action.detach().cpu().numpy()[0]
        action_clipped = np.clip(action_np, action_low, action_high)
        
        # Take step in environment
        next_state, reward, terminated, truncated, _ = env.step(action_clipped)
        done = terminated or truncated
        
        # Store transition
        states.append(state)
        actions.append(action_np)
        rewards.append(float(reward))
        log_probs.append(log_prob)
        
        state = next_state
        if done:
            break
    
    return states, actions, rewards, log_probs

# TRAINING FUNCTION
def train_reinforce(baseline_type):
    """
    Train REINFORCE with specified baseline using BATCH training.
    
    Args:
        baseline_type: one of ["no_baseline", "avg_reward", "reward_to_go", "value_function"]
    """
    print(f"\n{'='*70}")
    print(f"TRAINING: {baseline_type}")
    print(f"{'='*70}")
    
    # Create environment
    env = gym.make(ENV_NAME)
    env.reset(seed=SEED)
    set_seed(SEED, env)  # Seed both global RNG and environment
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Target reward: {TARGET_REWARD}")
    print(f"Max episodes: {NUM_EPISODES}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Initialize policy network
    policy = PolicyNetwork(state_dim, action_dim, HIDDEN_DIM).to(device)
    optimizer_policy = optim.Adam(policy.parameters(), lr=POLICY_LR, amsgrad=True)
    
    # Initialize value network if needed
    value_net = None
    optimizer_value = None
    if baseline_type == "value_function":
        value_net = ValueNetwork(state_dim, HIDDEN_DIM).to(device)
        optimizer_value = optim.Adam(value_net.parameters(), lr=VALUE_LR, amsgrad=True)
    
    # Tracking
    all_episode_rewards = []
    episode_rewards_deque = deque(maxlen=100)
    best_mean_reward = -np.inf
    best_model = None
    
    episode = 0
    pbar = tqdm(total=NUM_EPISODES, desc=f"Training {baseline_type}")
    
    # Training loop
    while episode < NUM_EPISODES:
        # 1. COLLECT BATCH OF EPISODES
        batch_states = []
        batch_actions = []
        batch_rewards = []  # List of episode reward lists
        batch_log_probs = []
        
        for _ in range(BATCH_SIZE):
            episode_states, episode_actions, episode_rewards, episode_log_probs = generate_episode(
                env, policy, action_low, action_high
            )
            
            batch_states.extend(episode_states)
            batch_actions.extend(episode_actions)
            batch_rewards.append(episode_rewards)  # Append whole episode
            batch_log_probs.extend(episode_log_probs)
            
            # Track episode return
            total_rewards = sum(episode_rewards)
            episode_rewards_deque.append(total_rewards)
            all_episode_rewards.append(total_rewards)
            episode += 1
            
            if episode >= NUM_EPISODES:
                break

        # 2. COMPUTE RETURNS/ADVANTAGES BASED ON BASELINE TYPE
        if baseline_type == "no_baseline":
            returns = compute_returns_no_baseline(batch_rewards, GAMMA)
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
            
        elif baseline_type == "reward_to_go":
            returns = compute_returns_reward_to_go(batch_rewards, GAMMA)
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
            
        elif baseline_type == "avg_reward":
            returns = compute_returns_avg_baseline(batch_rewards, GAMMA)
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
            
        elif baseline_type == "value_function":
            advantages, returns = compute_advantages_value_baseline(
                batch_rewards, batch_states, value_net, GAMMA, device
            )
            returns_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
            
        else:
            raise ValueError(f"Unknown baseline: {baseline_type}")
        
        # 3. UPDATE POLICY
        log_probs_tensor = torch.cat(batch_log_probs).to(device)
        
        # Check shapes match
        assert returns_tensor.shape == log_probs_tensor.shape, \
            f"Shape mismatch: returns {returns_tensor.shape} vs logs {log_probs_tensor.shape}"
        
        # Policy loss: -E[log π(a|s) * advantage]
        policy_loss = -(log_probs_tensor * returns_tensor).mean()
        
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()
        
        # 4. UPDATE VALUE NETWORK (if using value function baseline)
        if baseline_type == "value_function":
            returns_target = torch.tensor(returns, dtype=torch.float32, device=device)
            states_tensor = torch.tensor(np.array(batch_states, dtype=np.float32), 
                                        dtype=torch.float32, device=device)
            values = value_net(states_tensor)
            value_loss = F.mse_loss(values, returns_target)
            
            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()
        
        # 5. LOGGING
        avg_reward = np.mean(list(episode_rewards_deque)) if len(episode_rewards_deque) > 0 else 0.0
        
        # Track best model
        if avg_reward > best_mean_reward:
            best_mean_reward = avg_reward
            best_model = {
                'policy': policy.state_dict(),
                'value_net': value_net.state_dict() if value_net else None
            }
        
        if episode % 100 == 0:
            max_recent = np.max(all_episode_rewards[-BATCH_SIZE:]) if len(all_episode_rewards) >= BATCH_SIZE else 0
            log_msg = (f"Ep {episode}/{NUM_EPISODES} | "
                      f"Avg Reward: {avg_reward:7.2f} | "
                      f"Loss: {policy_loss.item():7.2f} | "
                      f"Max: {max_recent:4.0f}")
            pbar.write(log_msg)
        
        pbar.update(BATCH_SIZE)
        
        # 6. CHECK STOPPING CONDITION
        if avg_reward > 500.0:
            pbar.write(f"\nTarget reached! Avg reward: {avg_reward:.2f}")
            pbar.write(f"  Trained for {episode} episodes\n")
            break
    
    pbar.close()
    env.close()

    # Save model
    model_path = f"models/reinforce_{baseline}.pt"
    torch.save(best_model, model_path)
    print(f"Model saved → {model_path}")
            
    # Plot
    plot_training_curve(all_episode_rewards, baseline)
    print(f"Plot saved → plots/all_rewards_{baseline}.png")
    
    return all_episode_rewards, best_model

# MAIN EXECUTION
if __name__ == "__main__":
    logging(s="Q2_1")
    set_seed(SEED)
    
    print("\n" + "="*70)
    print("Q2 Step 1: Train REINFORCE with 4 Different Baselines")
    print("="*70)
    print(f"Environment: {ENV_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Target reward: {TARGET_REWARD}")
    print(f"Device: {device}")
    print("="*70 + "\n")
    
    baselines = [
        "no_baseline",
        "reward_to_go",
        "avg_reward",
        "value_function"
    ]
    
    for baseline in baselines:
        train_reinforce(baseline)