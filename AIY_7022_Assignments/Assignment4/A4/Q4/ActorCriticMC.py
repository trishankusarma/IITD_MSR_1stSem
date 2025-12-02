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
from collections import deque

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
SAVE_PATH = "models/a2c_lunar_lander_mc.pt"
ENTROPY_COEF = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("gifs", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# Logger helper
def logging(s="1"):
    global current_logger
    log_path = os.path.join("logs", f"output_{s}.txt")

    # If stdout is already a Logger, unwrap to real stdout
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


# SET SEED
def set_seed(env, seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except TypeError:
        pass


# ACTOR
class Actor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# CRITIC
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)  # [B]

def compute_mc_returns_and_advantages(rewards, values, gamma=GAMMA):
    """
    Monte-Carlo (reward-to-go) returns + advantages.
    returns[t] = sum_{k=t..T-1} gamma^{k-t} * rewards[k]
    advantages[t] = returns[t] - values[t]
    """
    T = len(rewards)
    returns = torch.zeros(T, dtype=torch.float32, device=device)
    
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G

    advantages = returns - values
    return returns, advantages

def compute_td_returns_and_advantages(rewards, dones, values, last_value, gamma=GAMMA):
    """
    1-step TD residual advantages (matches assignment eq. (1)):

    td_target[t] = r_t + gamma * V(s_{t+1})
    A_t         = td_target[t] - V(s_t)

    We do it in a batched way over the whole episode.
    """
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)   # [T]
    dones   = torch.tensor(dones,   dtype=torch.float32, device=device)   # [T], 1.0 if done else 0.0

    # values: [T] = V(s_t)
    # Build V(s_{t+1}) = [V(s_1), V(s_2), ..., V(s_{T-1}), last_value]
    next_values = torch.cat(
        [values[1:], torch.tensor([last_value], dtype=torch.float32, device=device)]
    )  # [T]

    td_target = rewards + gamma * next_values * (1.0 - dones)   # [T]
    advantages = td_target - values                             # [T]

    returns = td_target  # for critic regression

    return returns, advantages

# TRAIN
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

    actor_losses, critic_losses, total_losses = [], [], []

    best_avg_reward = -float("inf")
    best_episode = 0

    print("\nStarting training...\n")

    for ep in tqdm(range(1, MAX_EPISODES + 1), desc="Training Episodes"):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        step = 0

        states = []
        actions = []
        rewards = []
        dones = []

        # 1. Collect full episode
        while not done and step < MAX_STEPS_PER_EPISODE:
            s_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # Actor: π(a|s)
            logits = actor(s_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(float(done))

            ep_reward += reward
            state = next_state
            total_steps += 1
            step += 1

        # 2. Critic values for all states
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        values = critic(states_tensor)  # [T]

        # 3. Bootstrap last value
        if done:
            last_value = 0.0
        else:
            s_last = torch.FloatTensor(state).unsqueeze(0).to(device)
            last_value = critic(s_last).item()

        # 4. Compute returns & advantages (batched TD)
        # Monte-Carlo returns + advantages
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        returns, advantages = compute_mc_returns_and_advantages(rewards_tensor, values, gamma=GAMMA)

        # 4. Compute returns & advantages (batched 1-step TD)
        # returns, advantages = compute_td_returns_and_advantages(
        #     rewards, dones, values, last_value, gamma=GAMMA
        # )

        # Normalize advantages (important when using full-episode batch)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Clip advantages to prevent gradient explosion
        # advantages = torch.clamp(advantages, -5.0, 5.0)

        # 5. Actor loss
        logits_batch = actor(states_tensor)
        dist_batch = Categorical(logits=logits_batch)
        log_probs_batch = dist_batch.log_prob(actions_tensor)
        entropy = dist_batch.entropy().mean()

        actor_loss = -(log_probs_batch * advantages.detach()).mean() - ENTROPY_COEF * entropy

        # 6. Critic loss
        critic_loss = F.smooth_l1_loss(values, returns.detach())

        # 7. Backprop & update
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        total_loss = actor_loss + critic_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)

        actor_optimizer.step()
        critic_optimizer.step()

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())
        total_losses.append(total_loss.item())

        # 8. Logging and early stopping
        episode_rewards.append(ep_reward)
        running_avg_window.append(ep_reward)
        running_avg = np.mean(running_avg_window) if len(running_avg_window) > 0 else 0.0

        if running_avg > best_avg_reward and len(running_avg_window) >= 100:
            best_avg_reward = running_avg
            best_episode = ep
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "episode": ep,
                    "avg_reward": running_avg,
                },
                SAVE_PATH,
            )

        if ep % LOG_EVERY_EPISODES == 0:
            elapsed = time.time() - start_time
            recent_100 = (
                episode_rewards[-100:]
                if len(episode_rewards) >= 100
                else episode_rewards
            )
            print(f"\nEpisode {ep}/{MAX_EPISODES}")
            print(f"  Reward: {ep_reward:.1f}")
            print(f"  Avg(100): {running_avg:.2f}")
            print(f"  Std(100): {np.std(recent_100):.2f}")
            print(f"  Max(100): {np.max(recent_100):.2f}")
            print(f"  Steps: {total_steps}")
            print(f"  Time: {elapsed:.1f}s")
            if best_avg_reward > -float("inf"):
                print(f"  Best Avg: {best_avg_reward:.2f} (ep {best_episode})")

        if running_avg >= TARGET_RUNNING_AVG and len(running_avg_window) >= 100:
            print(f"\nSolved at episode {ep} with avg reward {running_avg:.2f}!")
            break

    env.close()
    print(f"\nTraining completed in {time.time() - start_time:.1f}s")
    print(
        f"Best model saved at episode {best_episode} "
        f"with avg reward {best_avg_reward:.2f}"
    )

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
    print("Actor-Critic Training for LunarLander-v3")
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
        f"with avg reward {checkpoint['avg_reward']:.2f}"
    )

    # Evaluate
    mean_reward, std_reward, eval_rewards = evaluate(
        actor, critic, num_episodes=5, render_gifs=True
    )