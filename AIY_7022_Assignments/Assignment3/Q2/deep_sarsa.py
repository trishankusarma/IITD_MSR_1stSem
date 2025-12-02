import os
import random
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import imageio
from utils.loggerUtils import Logger
from utils.replayBuffer import ReplayBuffer
import sys
import pandas as pd
import matplotlib.pyplot as plt

#  Logging & Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#  Hyperparameters 
SEED = 42
GAMMA = 0.99
BATCH_SIZE = 64               # training batch size from temporary buffer
NUM_ITERATIONS = 7000        # number of episodes (outer loop)
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.9995            # per-episode decay
EVAL_EPISODES_PER_CHECK = 10
FINAL_EVAL_EPISODES = 100
TARGET_UPDATE_FREQ = 1000     # steps (global steps)
MAX_EPISODE_LEN = 1000
initial_LR = 1e-3
final_LR = 1e-3
warmup_iters = 2000

# replay buffer size to avoid memory blow-up
MAX_REPLAY_SIZE = 100_000

# Paths 
MODEL_PATH = 'models/best_deep_sarsa.pt'  # change to Q2 paths if required by grader
EVAL_JSON = 'evaluation/lunarlander_evaluation_results.json'
GIF_PATH = 'gifs/lunarlander.gif'

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(EVAL_JSON), exist_ok=True)
os.makedirs(os.path.dirname(GIF_PATH), exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

original_stdout = sys.stdout


from datetime import datetime

def logging(s='1'):
    global current_logger
    
    log_path = os.path.join('logs', f'output_{s}.txt')
    current_logger = Logger(log_path)
    sys.stdout = current_logger

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("="*70)
    print(f"Logging started at {timestamp}")
    print(f"Log file   : {log_path}")
    print(f"Using device: {device}")
    print("="*70)


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


set_seed(SEED)  # global RNG seeded; don't reseed env repeatedly during training

def get_warmup_lr(iteration):
    if iteration >= warmup_iters:
        return final_LR
    return initial_LR + (final_LR - initial_LR) * (iteration / warmup_iters)

# Neural Network 
class NonLinearQ(nn.Module):
    def __init__(self, input_dim=8, output_dim=4, h1=256, h2=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

#  Deep SARSA Agent 
class DeepSARSAAgent:
    def __init__(self, env, seed=42, device = "cuda", gamma = 0.99):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        self.net = NonLinearQ(input_dim=self.obs_dim, output_dim=self.act_dim).to(device)
        self.target_net = copy.deepcopy(self.net).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=initial_LR)

        self.eps = EPS_START
        self.gamma = gamma
        self.totalSteps = 0
        self.device = device

    def select_action(self, state, greedy=False):
        if (not greedy) and (random.random() < self.eps):
            return self.env.action_space.sample()
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q = self.net(t)
            return int(q.argmax(dim=1).item())

    def update_from_buffer(self, buffer: ReplayBuffer):
        """
        Single gradient update using a minibatch sampled from buffer.
        Returns loss scalar or None if not enough data.
        """
        if len(buffer) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, next_actions, dones = buffer.sample(BATCH_SIZE)
        states, actions = states.to(self.device), actions.to(self.device)
        rewards, next_states = rewards.to(self.device), next_states.to(self.device)
        next_actions, dones = next_actions.to(self.device), dones.to(self.device)

        q_sa = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + (1.0 - dones) * self.gamma * q_next

        loss = F.mse_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.net.state_dict())

    def evaluate(self, num_episodes=10, greedy=True):
        old_eps = self.eps
        self.eps = 0.0 if greedy else self.eps
        rewards = []
        for idx in range(num_episodes):
            obs, _ = self.env.reset(seed=SEED + idx)  # deterministic evaluation
            total, step = 0.0, 0
            a = self.select_action(obs, greedy=True)
            done_flag = False
            while not done_flag and step < MAX_EPISODE_LEN:
                next_obs, r, terminated, truncated, info = self.env.step(a)
                total += r
                done_flag = terminated or truncated
                next_a = self.select_action(next_obs, greedy=True)
                obs, a = next_obs, next_a
                step += 1
            rewards.append(total)
        self.eps = old_eps
        return np.mean(rewards), np.std(rewards), np.max(rewards), np.min(rewards)

#  GIF 
def generate_gif(env, agent, filename, max_steps=MAX_EPISODE_LEN):
    frames = []
    obs, _ = env.reset(seed=SEED)  # deterministic gif
    done, step = False, 0
    a = agent.select_action(obs, greedy=True)
    while not done and step < max_steps:
        frame = env.render()
        frames.append(frame)
        next_obs, r, terminated, truncated, info = env.step(a)
        next_a = agent.select_action(next_obs, greedy=True)
        obs, a = next_obs, next_a
        done = terminated or truncated
        step += 1
    imageio.mimsave(filename, frames, fps=30)


#  Training Loop 
def train_loop():
    env = gym.make('LunarLander-v3')
    eval_env = gym.make('LunarLander-v3')
    agent = DeepSARSAAgent(env, seed=SEED, device=device)
    best_avg = -float('inf')
    best_state = None
    episode_rewards = []
    episode_losses = []

    replay = ReplayBuffer(max_size=MAX_REPLAY_SIZE)

    for itr in range(1, NUM_ITERATIONS + 1):
        # set LR warmup
        lr = get_warmup_lr(itr)
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = lr

        obs, _ = env.reset()  # do not pass seed → stochastic starts but reproducible via global RNG
        a = agent.select_action(obs)
        ep_reward = 0.0
        ep_losses = []

        for step in range(MAX_EPISODE_LEN):
            next_obs, r, terminated, truncated, info = env.step(a)
            done_flag = terminated or truncated
            next_a = agent.select_action(next_obs)

            # push transition and do one update (on-policy style; batch sampling for stability)
            replay.push(obs.astype(np.float32), a, float(r), next_obs.astype(np.float32), next_a, float(done_flag))

            # increment global step counter for target update scheduling
            agent.totalSteps += 1

            if len(replay) >= BATCH_SIZE:
                loss_val = agent.update_from_buffer(replay)
                if loss_val is not None:
                    ep_losses.append(loss_val)

            obs, a = next_obs, next_a
            ep_reward += r

            # target network update by global step
            if agent.totalSteps % TARGET_UPDATE_FREQ == 0:
                agent.target_net.load_state_dict(agent.net.state_dict())

            if done_flag:
                break

        # episode end
        episode_rewards.append(ep_reward)
        episode_losses.append(np.mean(ep_losses) if len(ep_losses) > 0 else 0.0)

        # epsilon decay per episode
        agent.eps = max(EPS_END, agent.eps * EPS_DECAY)

        # periodic evaluation & saving best
        if itr % 10 == 0:
            mean_r, std_r, maxm, minm = agent.evaluate(num_episodes=EVAL_EPISODES_PER_CHECK)
            print(f"Iter {itr} | Eval mean = {mean_r:.2f} | Max reward = {maxm:.2f} | Min reward = {minm:.2f} | eps = {agent.eps:.3f}")
            if mean_r > best_avg:
                best_avg = mean_r
                best_state = copy.deepcopy(agent.net.state_dict())
                agent.save(MODEL_PATH)
                print(f"New best model saved (mean {best_avg:.2f}) at iter {itr}")

        # periodic logging of progress
        if itr % 500 == 0:
            recent_rewards = episode_rewards[-500:] if len(episode_rewards) >= 500 else episode_rewards
            recent_losses = episode_losses[-500:] if len(episode_losses) >= 500 else episode_losses
            print(f"--> Iter {itr}/{NUM_ITERATIONS} | AvgReward(last {len(recent_rewards)}): {np.mean(recent_rewards):.2f} | "
                  f"AvgLoss: {np.mean(recent_losses):.6f} | Eps: {agent.eps:.3f} | ReplayLen: {len(replay)}")

    env.close()
    eval_env.close()
    print(f"Training complete. Best mean reward: {best_avg:.2f}")
    return episode_rewards


#  Plotting 
def plot_rewards(episode_rewards, window):
    smoothed_rewards = pd.Series(episode_rewards).rolling(window=window).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(smoothed_rewards, label=f'Rolling Mean (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards per Episode')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/episode_rewards.png')
    plt.show()


#  Final Evaluation 
def final_evaluation_and_gif():
    eval_env = gym.make('LunarLander-v3')
    gif_env = gym.make('LunarLander-v3', render_mode='rgb_array')

    agent = DeepSARSAAgent(eval_env, seed=SEED, device = device)
    agent.load(MODEL_PATH)

    mean_r, std_r, maxm, minm = agent.evaluate(num_episodes=FINAL_EVAL_EPISODES)
    results = {'mean': float(mean_r), 'std': float(std_r), 'max': float(maxm), 'min': float(minm)}
    os.makedirs(os.path.dirname(EVAL_JSON), exist_ok=True)
    with open(EVAL_JSON, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Final evaluation saved: mean={mean_r:.2f}, std={std_r:.2f}, max={maxm:.2f}, min={minm:.2f}")

    generate_gif(gif_env, agent, GIF_PATH)
    print(f"GIF saved to {GIF_PATH}")

    eval_env.close()
    gif_env.close()


#  Entrypoint 
if __name__ == '__main__':
    logging(s="q2")
    print('Starting Deep SARSA training for LunarLander-v3 (one update per step)')
    episode_rewards = train_loop()
    plot_rewards(episode_rewards, 10)
    final_evaluation_and_gif()
    print('All done.')
