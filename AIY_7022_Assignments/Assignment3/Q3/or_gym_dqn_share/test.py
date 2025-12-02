import sys
import gymnasium as gym
sys.modules["gym"] = gym  

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.envs.registration")
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import matplotlib.pyplot as plt
import os
import json
from tqdm import trange
import random
from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv


# -------------------- PRIORITIZED EXPERIENCE REPLAY BUFFER --------------------
class PER_Buffer:
    def __init__(self, max_size=100000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = int(max_size)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.frame = 1

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        data = (state.copy(), int(action), float(reward), next_state.copy(), bool(done))
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]

        probs = prios ** self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / max(1, self.beta_frames)))
        self.frame += 1
        total = len(self.buffer)

        weights = (total * probs[indices]) ** (-beta)
        weights = weights / (weights.max() + 1e-8)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.array(dones, dtype=np.uint8),
            indices,
            np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio + 1e-6

    def __len__(self):
        return len(self.buffer)


# -------------------- DQN NETWORK --------------------
class DQN(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden=[128, 64]):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.out = nn.Linear(hidden[1], num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


# -------------------- AGENT --------------------
class Agent:
    def __init__(self, env, max_buffer_len=100000, min_buffer_len=2000, num_episodes=5000, batch_size=64,
                 gamma=0.99, lr=1e-4, target_update_freq=1000, epsilon_start=1.0, epsilon_end=0.01, part='a',
                 device=None):

        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.env = env
        self.min_buffer_len = min_buffer_len
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.total_steps = 0
        self.part = part

        lot = 2
        num_assets = env.action_space.shape[0]
        act_vals = list(range(-lot, lot + 1))
        from itertools import product
        self.action_list = [np.array(a, dtype=np.int32) for a in product(act_vals, repeat=num_assets)]
        self.n_actions = len(self.action_list)
        self.obs_dim = env.reset().shape[0]

        self.current_net = DQN(self.obs_dim, self.n_actions).to(self.device)
        self.target_net = DQN(self.obs_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.current_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.current_net.parameters(), lr=lr)
        alpha = 0.7 if part == 'a' else 0.6
        self.buffer = PER_Buffer(max_size=max_buffer_len, alpha=alpha)
        self.losses = []
        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.lr = lr

    def _action_from_index(self, idx):
        return self.action_list[int(idx)]

    def is_terminal_from_state(self, state):
        step = int(state[-1])
        holdings = state[1 + self.env.num_assets:1 + self.env.num_assets + self.env.num_assets]
        if step >= self.env.step_limit or np.any(holdings < 0):
            return True
        return False

    def initial_buffer_fill(self, steps_to_fill=None):
        print("initial populating replay buffer...")
        while len(self.buffer) < (self.min_buffer_len if steps_to_fill is None else steps_to_fill):
            s = self.env.reset()
            done = False
            while not done:
                action_idx = random.randrange(self.n_actions)
                action = self._action_from_index(action_idx)
                next_s, r, done, _ = self.env.step(action)
                self.buffer.push(s, action_idx, r, next_s, done)
                s = next_s
            if len(self.buffer) >= (self.min_buffer_len if steps_to_fill is None else steps_to_fill):
                break
    
    def rollout_greedy(self, env, seed=None, render=False):
        # pure greedy rollout for eval
        set_seed(seed)
        s = env.reset()
        wealths = []
        done = False
        step_count = 0
        max_steps = env.step_limit + 1
        while not done and step_count < max_steps:
            cash = float(s[0])
            prices = s[1:1 + env.num_assets]
            holdings = s[1 + env.num_assets: 1 + env.num_assets + env.num_assets]
            wealth = cash + float(np.dot(prices, holdings))
            wealths.append(wealth)

            with torch.no_grad():
                s_t = torch.tensor(s.astype(np.float32)).unsqueeze(0).to(self.device)
                q_values = self.current_net(s_t)
                action_idx = int(q_values.argmax(1).item())
            action = self._action_from_index(action_idx)
            next_s, r, done, _ = env.step(action)
            s = next_s
            step_count += 1

        if (len(wealths) < env.step_limit) and done:
            # compute final wealth and append
            cash = float(s[0])
            prices = s[1:1 + env.num_assets]
            holdings = s[1 + env.num_assets: 1 + env.num_assets + env.num_assets]
            wealth = cash + float(np.dot(prices, holdings))
            wealths.append(wealth)

        if len(wealths) < env.step_limit:
            if len(wealths) == 0:
                wealths = [0.0] * env.step_limit
            else:
                last = wealths[-1]
                wealths += [last] * (env.step_limit - len(wealths))
        return np.array(wealths[:env.step_limit], dtype=np.float32)
    
    def evaluate_and_plot(self, seeds=100, outdir="results"):
        # eval over multiple seeds and plot mean ± std across timesteps
        os.makedirs(outdir, exist_ok=True)
        all_wealths = []
        eval_env = DiscretePortfolioOptEnv()

        for s in range(seeds):
            w = self.rollout_greedy(eval_env, seed=s)
            all_wealths.append(w)
        all_wealths = np.vstack(all_wealths)

        mean_wealth = np.mean(all_wealths, axis=0)
        std_wealth = np.std(all_wealths, axis=0)

        # plot mean with shaded std
        plt.figure(figsize=(8, 5))
        steps = np.arange(1, len(mean_wealth) + 1)
        plt.plot(steps, mean_wealth, label="Mean wealth")
        plt.fill_between(steps, mean_wealth - std_wealth, mean_wealth + std_wealth, alpha=0.3, label="±1 std")
        plt.xlabel("Timestep")
        plt.ylabel("Portfolio wealth")
        plt.title(f"Mean portfolio wealth over {seeds} seeds")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(outdir, f"mean_std_wealth_{self.part}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved wealth plot to {plot_path}")

        if len(self.losses) > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(np.arange(len(self.losses)), self.losses)
            plt.xlabel("Optimization steps")
            plt.ylabel("Loss (MSE)")
            plt.title("Training loss curve")
            plt.grid(True)
            loss_path = os.path.join(outdir, f"loss_curve_{self.part}.png")
            plt.tight_layout()
            plt.savefig(loss_path)
            plt.close()
            print(f"Saved loss plot to {loss_path}")

        mean_terminal = mean_wealth[-1]
        std_terminal = std_wealth[-1]
        ratio = mean_terminal / (std_terminal + 1e-12)


        results_dict = {
            "all_wealths": all_wealths.tolist(),
            "mean_wealth": mean_wealth.tolist(),
            "std_wealth": std_wealth.tolist(),
            "mean_terminal": float(mean_terminal),
            "std_terminal": float(std_terminal),
            "ratio": float(ratio)
        }
       
        json_path = os.path.join(outdir, f"wealth_stats_{self.part}.json")
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=4)
       

        return results_dict

    def train(self):
        all_rewards = []
        opt_steps = 0
        for episode in trange(self.num_episodes, desc="Episodes"):
            s = self.env.reset()
            ep_reward = 0.0
            done = False
            steps_ep = 0
            eps = max(self.eps_end, self.eps_start * ((self.eps_end / self.eps_start) ** (episode / self.num_episodes)))

            while not done:
                if random.random() < eps:
                    action_idx = random.randrange(self.n_actions)
                else:
                    with torch.no_grad():
                        s_t = torch.tensor(s.astype(np.float32)).unsqueeze(0).to(self.device)
                        q_values = self.current_net(s_t)
                        action_idx = int(q_values.argmax(1).item())

                action = self._action_from_index(action_idx)
                cash = float(s[0])
                prices = s[1:1 + self.env.num_assets]
                holdings = s[1 + self.env.num_assets:1 + 2 * self.env.num_assets]
                prev_value = cash + np.dot(prices, holdings)

                next_s, r_env, done, _ = self.env.step(action)
                cash_next = float(next_s[0])
                prices_next = next_s[1:1 + self.env.num_assets]
                holdings_next = next_s[1 + self.env.num_assets:1 + 2 * self.env.num_assets]
                current_value = cash_next + np.dot(prices_next, holdings_next)

                if self.part == 'b':
                    r_shaped = np.log((current_value + 1e-8) / (prev_value + 1e-8)) - 0.5 * (np.log((current_value + 1e-8) / (prev_value + 1e-8))) ** 2
                else:
                    r_shaped = r_env if done else 0

                self.buffer.push(s, action_idx, r_shaped, next_s, done)
                ep_reward += r_env
                s = next_s
                steps_ep += 1
                self.total_steps += 1

                if len(self.buffer) >= self.batch_size:
                    self.update_model()
                    opt_steps += 1

                if self.total_steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.current_net.state_dict())

                if steps_ep > 500:
                    break

            all_rewards.append(ep_reward)
            if (episode + 1) % 50 == 0:
                avg_r = np.mean(all_rewards[-50:])
                print(f"Ep {episode + 1}: Avg Reward (50ep)={avg_r:.3f}, ε={eps:.3f}")
        return all_rewards

    def update_model(self):
        states, actions, rewards, next_states, dones, idxs, weights = self.buffer.sample(self.batch_size)
        states_t = torch.tensor(states).float().to(self.device)
        next_states_t = torch.tensor(next_states).float().to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        weights_t = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.current_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_actions = self.target_net(next_states_t).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        td_errors = (target_q - q_values).detach().cpu().numpy().squeeze()
        loss = (weights_t * F.mse_loss(q_values, target_q, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.current_net.parameters(), 10.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        self.buffer.update_priorities(idxs, np.abs(td_errors) + 1e-6)

    def save(self, path="models"):
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, f"per_ddqn_portfolio_{self.part}.pt")
        torch.save(self.current_net.state_dict(), model_path)
        print(f"Saved model to {model_path}")
        return model_path
    
# ----------------------------
# Utility: Set global seed
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ----------------------------
# Run Training
# ----------------------------
def run_training(part='a', num_episodes=5000, seed=42):
    print(f"\n{'='*30}")
    print(f"Training DQN Portfolio Agent (part={part})")
    print(f"{'='*30}\n")

    set_seed(seed)
    env = DiscretePortfolioOptEnv(env_config={"seed": seed})

    agent = Agent(
        env=env,
        max_buffer_len=100000,
        min_buffer_len=2000,
        num_episodes=num_episodes,
        batch_size=64,
        gamma=0.99,
        lr=1e-4,
        target_update_freq=1000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        part=part,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Fill initial buffer
    agent.initial_buffer_fill()

    # Train
    rewards = agent.train()

    # Save model
    model_path = agent.save(path="models")

    # Evaluate
    stats = agent.evaluate_and_plot(seeds=100, outdir="results")

    # Plot training reward curve
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(rewards)), rewards)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title(f"Training Reward Curve (part {part})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("results", f"training_rewards_{part}.png"))
    plt.close()

    print(f"Training complete for part {part}")
    print(f"Model saved at: {model_path}")
    print(f"Results saved under: results/wealth_stats_{part}.json")
    print(f"Final mean wealth: {stats['mean_terminal']:.3f}")
    print(f"Std: {stats['std_terminal']:.3f}")
    print(f"Mean/Std ratio: {stats['ratio']:.3f}")

    return stats


# ----------------------------
# Main Entry
# ----------------------------
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Task 1: Maximize Terminal Wealth
    # stats_a = run_training(part='a', num_episodes=5000, seed=42)

    # Task 2: Maximize Wealth across all timesteps
    stats_b = run_training(part='b', num_episodes=20000, seed=42)

    print("\nSummary:")
    print(f"Part (a) - Terminal Wealth: Mean={stats_a['mean_terminal']:.3f}, "
          f"Std={stats_a['std_terminal']:.3f}, Ratio={stats_a['ratio']:.3f}")
    print(f"Part (b) - Step Wealth: Mean={stats_b['mean_terminal']:.3f}, "
          f"Std={stats_b['std_terminal']:.3f}, Ratio={stats_b['ratio']:.3f}")