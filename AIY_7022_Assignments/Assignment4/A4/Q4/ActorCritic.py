"""
Starter Code for Q4
ActorCritic.py

Usage:
    python ActorCritic.py

Requirements:
    pip install gymnasium torch numpy matplotlib box2d
"""
import math
import time
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from utils.plotUtils import generatePlots, save_gif
from utils.loggerUtils import Logger
import os
from tqdm import tqdm
import sys
from datetime import datetime

# Hyperparameters
ENV_NAME = "LunarLander-v3"
SEED = 42
HIDDEN_SIZE = 128
LR_ACTOR = 1e-3  
LR_CRITIC = 1e-3 
GAMMA = 0.99
MAX_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 1000
TARGET_RUNNING_AVG = 250.0
LOG_EVERY_EPISODES = 100
SAVE_PATH = "models/a2c_lunar_lander.pt"
ENTROPY_COEF = 0.001 
MAX_GRAD_NORM = 0.5
device = "cpu"

# Directories
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("gifs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

Transition = namedtuple("Transition", ("state", "action", "log_prob", "reward", "done", "value"))

def logging(s="1"):
    global current_logger
    log_path = os.path.join("logs", f"output_{s}.txt")

    if isinstance(sys.stdout, Logger):
        sys.stdout = sys.stdout.terminal

    current_logger = Logger(log_path)
    sys.stdout = current_logger

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"Logging started at {timestamp}")
    print(f"Log file   : {log_path}")
    print(f"Using device: {device}")
    print("=" * 70)


def set_seed(env, seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except TypeError:
        pass


# Actor Network
class Actor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Critic Network
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


def compute_returns_and_advantages_mc(transitions, last_value, gamma=GAMMA):
    """
    Monte Carlo returns
    
    Returns: G_t = sum_{k=t}^T gamma^(k-t) * r_k
    Advantages: A_t = G_t - V(s_t)
    """
    rewards = [t.reward for t in transitions]
    dones = [t.done for t in transitions]
    values = [t.value for t in transitions]
    
    # Compute MC returns
    returns = []
    G = last_value
    
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            G = r
        else:
            G = r + gamma * G
        returns.insert(0, G)
    
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    
    # Compute advantages using TD residual with MC returns
    advantages = []
    for i in range(len(transitions)):
        v = values[i]
        if transitions[i].done:
            td_target = rewards[i]
        else:
            if i < len(transitions) - 1:
                td_target = rewards[i] + gamma * values[i + 1]
            else:
                td_target = rewards[i] + gamma * last_value
        
        advantage = td_target - v
        advantages.append(advantage)
    
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    
    return returns, advantages


def train():
    env = gym.make(ENV_NAME)
    set_seed(env, SEED)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    actor = Actor(obs_dim, n_actions, HIDDEN_SIZE).to(device)
    critic = Critic(obs_dim, HIDDEN_SIZE).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    print("\nActor Network:")
    for name, param in actor.named_parameters():
        print(f"  {name}: {param.shape}")

    print("\nCritic Network:")
    for name, param in critic.named_parameters():
        print(f"  {name}: {param.shape}")

    episode_rewards = []
    running_avg_window = deque(maxlen=100)
    total_steps = 0
    start_time = time.time()

    actor_losses, critic_losses = [], []
    entropies = []

    best_running_avg = -float("inf")
    best_episode = 0

    print("\nStarting training...\n")

    for ep in tqdm(range(1, MAX_EPISODES + 1), desc="Training Episodes"):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        step = 0

        transitions = []

        # Collect full episode
        for t in range(MAX_STEPS_PER_EPISODE):
            total_steps += 1
            
            # Get state tensor
            s_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            # Critic forward pass
            value = critic(s_tensor).squeeze(0)
            
            # Actor forward pass
            logits = actor(s_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action).squeeze(0)
            
            # Environment step
            ns, r, te, tr, _ = env.step(action.item())
            done = te or tr
            
            transitions.append(Transition(
                state=state,
                action=action.item(),
                log_prob=log_prob,
                reward=r,
                done=done,
                value=value.item()
            ))
            
            ep_reward += r
            state = ns
            step += 1
            
            if done:
                break
        
        # Bootstrap last value
        if done:
            last_value = 0.0
        else:
            next_state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                last_value = critic(next_state_tensor).item()
        
        # Compute returns and advantages using MC
        returns, advantages = compute_returns_and_advantages_mc(
            transitions, last_value, gamma=GAMMA
        )
        
        # Prepare tensors
        states = torch.tensor([t.state for t in transitions], dtype=torch.float32, device=device)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=device)
        
        # Normalize advantages
        advantages_mean = advantages.mean()
        advantages_std = advantages.std()
        if advantages_std > 1e-8:
            advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
        
        # Clip advantages
        advantages = torch.clamp(advantages, -10.0, 10.0)
        
        # Actor loss
        logits_batch = actor(states)
        probs_batch = F.softmax(logits_batch, dim=-1)
        dist_batch = Categorical(probs_batch)
        log_probs_batch = dist_batch.log_prob(actions)
        entropy = dist_batch.entropy().mean()
        
        actor_loss = -(log_probs_batch * advantages.detach()).mean() - ENTROPY_COEF * entropy
        
        # Critic loss (regress to MC returns)
        values = critic(states)
        critic_loss = F.mse_loss(values, returns.detach())
        
        # SEPARATE optimization
        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
        actor_optimizer.step()
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
        critic_optimizer.step()
        
        # Track metrics
        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())
        entropies.append(entropy.item())
        
        # Episode tracking
        episode_rewards.append(ep_reward)
        running_avg_window.append(ep_reward)
        running_avg = np.mean(running_avg_window) if len(running_avg_window) > 0 else 0.0
        
        # Save best model
        if running_avg > best_running_avg and len(running_avg_window) >= 10:
            best_running_avg = running_avg
            best_episode = ep
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "episode": ep,
                    "running_avg": running_avg,
                },
                SAVE_PATH,
            )
        
        # Logging
        if ep % LOG_EVERY_EPISODES == 0:
            elapsed = time.time() - start_time
            recent_100 = (
                episode_rewards[-100:]
                if len(episode_rewards) >= 100
                else episode_rewards
            )
            
            recent_entropy = np.mean(entropies[-100:]) if len(entropies) >= 100 else np.mean(entropies)
            
            print(f"\nEpisode {ep}/{MAX_EPISODES}")
            print(f"  Reward: {ep_reward:.1f}")
            print(f"  Avg(100): {running_avg:.2f}")
            print(f"  Std(100): {np.std(recent_100):.2f}")
            print(f"  Max(100): {np.max(recent_100):.2f}")
            print(f"  Entropy: {recent_entropy:.4f}")
            print(f"  Actor Loss: {np.mean(actor_losses[-100:]):.4f}")
            print(f"  Critic Loss: {np.mean(critic_losses[-100:]):.4f}")
            print(f"  Steps: {total_steps}")
            print(f"  Time: {elapsed:.1f}s")
            if best_running_avg > -float("inf"):
                print(f"Best Avg: {best_running_avg:.2f} (ep {best_episode})")
            
            if recent_entropy < 0.1:
                print("Low entropy - policy may have collapsed!")
        
        # Early stopping
        if running_avg >= TARGET_RUNNING_AVG and len(running_avg_window) >= 100:
            print(f"\nSolved at episode {ep} with avg reward {running_avg:.2f}!")
            break
    
    env.close()
    print(f"\nTraining completed in {time.time() - start_time:.1f}s")
    print(f"Best model saved at episode {best_episode} with avg reward {best_running_avg:.2f}")
    
    generatePlots(actor_losses, critic_losses, episode_rewards)
    
    return actor, critic

def evaluate(actor, critic, num_episodes=5, render_gifs=True):
    """Evaluate the trained actor-critic model"""
    env = gym.make(ENV_NAME, render_mode="rgb_array" if render_gifs else None)
    
    actor.eval()
    critic.eval()
    
    eval_rewards = []
    
    for ep in range(num_episodes):
        state, _ = env.reset(seed=SEED + ep)
        done = False
        ep_reward = 0.0
        step = 0
        frames = []
        
        while not done and step < MAX_STEPS_PER_EPISODE:
            if render_gifs:
                frames.append(env.render())
            
            # Greedy action from actor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = actor(state_tensor)
                action = torch.argmax(logits, dim=1).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = next_state
            step += 1
        
        eval_rewards.append(ep_reward)
        print(f"Evaluation Episode {ep + 1}: Reward = {ep_reward:.2f}")
        
        if render_gifs and len(frames) > 0:
            save_gif(frames, f"gifs/a2c_episode_{ep + 1}.gif")
    
    env.close()
    
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    
    print("\nEvaluation Results:")
    print(f"  Mean Reward: {mean_reward:.2f}")
    print(f"  Std Reward: {std_reward:.2f}")
    print(f"  Max Reward: {np.max(eval_rewards):.2f}")
    print(f"  Min Reward: {np.min(eval_rewards):.2f}")
    
    return mean_reward, std_reward, eval_rewards


if __name__ == "__main__":
    logging("Q4")
    print("=" * 70)
    print("Optimized Actor-Critic Training for LunarLander-v3")
    print("=" * 70)
    
    # Train
    actor, critic = train()
    
    # Load best model and evaluate
    print("\n" + "=" * 70)
    print("Loading best model for evaluation...")
    print("=" * 70)
    
    actor = Actor(8, 4, HIDDEN_SIZE).to(device)
    critic = Critic(8, HIDDEN_SIZE).to(device)
    
    checkpoint = torch.load(SAVE_PATH, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["actor"])
    critic.load_state_dict(checkpoint["critic"])
    print(
        f"Loaded model from episode {checkpoint['episode']} "
        f"with running avg {checkpoint['running_avg']:.2f}"
    )
    
    # Evaluate
    mean_reward, std_reward, eval_rewards = evaluate(
        actor, critic, num_episodes=5, render_gifs=True
    )