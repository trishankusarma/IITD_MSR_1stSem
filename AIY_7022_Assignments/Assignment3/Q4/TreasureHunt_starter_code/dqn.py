import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.replayBuffer import PrioritizedReplayBuffer
import time
from tqdm import tqdm
import os
from env import TreasureHunt_v2
import random
from utils.loggerUtils import Logger
from utils.plotUtils import plot_training_rewards, plot_evaluation_curves
from datetime import datetime
import threading
import sys
import gc

# Hyperparameters
GAMMA = 0.99
MAX_BUFFER_SIZE = 100_000
MIN_BUFFER_SIZE = 5000
NUM_EPISODE_LEN = 50000
FINAL_EVAL_EPISODES = 100
LR = 1e-4
BATCH_SIZE = 64
EPS_START = 1.0
EPS_END = 0.05
DECAY_FACTOR = (EPS_END / EPS_START) ** (1.0 / NUM_EPISODE_LEN)
TARGET_UPDATE_FREQ = 1000
MAX_EPISODE_LENGTH = 100
SEED = 42

# Paths
MODEL_DIR = 'models'
GIF_DIR = 'gifs'
PLOT_DIR = 'plots'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GIF_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

original_stdout = None

# Utility Functions
def logging(s='1'):
    """Initialize timestamped logging."""
    global current_logger, original_stdout
    import sys
    original_stdout = sys.stdout

    log_path = os.path.join('logs', f'output_{s}.txt')
    current_logger = Logger(log_path)
    sys.stdout = current_logger

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"Logging started at {timestamp}")
    print(f"Log file   : {log_path}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 70)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Q-Network Definition
class QNetwork(nn.Module):
    def __init__(self, in_channels=4, num_actions=4):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 9, 64)
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# DQN Agent
class DQNAgent:
    def __init__(self,
                 state_shape=(4, 10, 10),
                 num_actions=4,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 gamma=GAMMA,
                 lr=LR,
                 buffer_capacity=MAX_BUFFER_SIZE,
                 batch_size=BATCH_SIZE,
                 epsilon_start=EPS_START,
                 epsilon_end=EPS_END,
                 decay_factor=DECAY_FACTOR,
                 target_update_freq=TARGET_UPDATE_FREQ):
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_factor = decay_factor
        self.target_update_freq = target_update_freq
        self.eval_mean_rewards = []
        self.eval_mean_treasures = []
        self.eval_mean_pirates = []
        self.eval_mean_goals = []

        # Networks
        self.policy_net = QNetwork().to(device)
        self.target_net = QNetwork().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity, state_shape=state_shape, device=device
        )

        self.epsilon = epsilon_start
        self.step_count = 0

    def is_terminal(self, spatial_state):
        ship_layer = spatial_state[3]
        return bool(ship_layer[9, 9])

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.policy_net(s).argmax(dim=1).item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(self.batch_size)
        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions)
            targets = rewards + self.gamma * (1 - dones) * next_q

        td_errors = q_values - targets
        per_sample_loss = F.smooth_l1_loss(q_values, targets, reduction='none')
        loss = (per_sample_loss * weights.unsqueeze(1)).mean()

        self.replay_buffer.update_priorities(indices, td_errors.detach().squeeze())

        if self.step_count > 0 and self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.step_count += 1

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.decay_factor, self.epsilon_end)

# Warmup & Evaluation
def warmup_replay_buffer(env, agent, min_size=MIN_BUFFER_SIZE):
    print(f"Filling replay buffer with {min_size} random transitions...")
    while len(agent.replay_buffer) < min_size:
        s = env.reset()
        for _ in range(100):
            a = np.random.randint(agent.num_actions)
            ns, r = env.step(a)
            done = agent.is_terminal(ns)
            agent.replay_buffer.push(s, a, r, ns, done)
            s = ns
            if len(agent.replay_buffer) >= min_size:
                break

def evaluate(agent, env, n_episodes=100, max_steps=100, seed_base=0):
    rewards, treasures, pirates, goals = [], [], [], []
    agent.policy_net.eval()

    for ep in range(n_episodes):
        set_seed(seed_base + ep)
        s = np.array(env.reset(), dtype=np.float32)
        total_r, treasure_hits, pirate_hits, goal_hits = 0, 0, 0, 0

        for step in range(max_steps):
            with torch.no_grad():
                s_t = torch.tensor(s, dtype=torch.float32, device=agent.device).unsqueeze(0)
                a = agent.policy_net(s_t).argmax(1).item()

            ns, r = env.step(a)
            ns = np.array(ns, dtype=np.float32)
            total_r += r
            done = agent.is_terminal(ns)

            pos = np.argwhere(ns[3] == 1)
            if len(pos) > 0:
                r_, c_ = pos[0]
                if ns[2, r_, c_] == 1:
                    treasure_hits += 1
                elif ns[1, r_, c_] == 1:
                    pirate_hits += 1
                elif (r_, c_) == (9, 9) and done:
                    goal_hits += 1

            if done:
                break
            s = ns

        rewards.append(total_r)
        treasures.append(treasure_hits)
        pirates.append(pirate_hits)
        goals.append(goal_hits)

    agent.policy_net.train()

    print(
        f"Eval → Reward={np.mean(rewards):.2f}, Treasures={np.mean(treasures):.2f}, "
        f"Pirates={np.mean(pirates):.2f}, Goals={np.mean(goals):.2f}"
    )

    agent.eval_mean_rewards.append(np.mean(rewards))
    agent.eval_mean_treasures.append(np.mean(treasures))
    agent.eval_mean_pirates.append(np.mean(pirates))
    agent.eval_mean_goals.append(np.mean(goals))

# Visualization
def visualiseModel(agent, env, rewards_per_episode,
                   model_name=None, gif_dir=None, plot_dir=None, model_dir=None,
                   random_policy_gif_needed=False, is_time_stamp_needed = True):
    
    if is_time_stamp_needed:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_name}_{timestamp}"
    if model_dir is not None:
        SAVE_MODEL_PATH = os.path.join(model_dir, f"{model_name}.pt")
        torch.save(agent.policy_net.state_dict(), SAVE_MODEL_PATH)
        print(f"Model saved to {SAVE_MODEL_PATH}")

    if plot_dir is not None:
        plot_training_rewards(plot_dir, model_name, rewards_per_episode)

    if gif_dir is not None:
        TRAINED_GIF_PATH = os.path.join(gif_dir, f"{model_name}_trained.gif")
        all_states = env.get_all_states().astype(np.float32)  # cast once
        batch_size = 512
        Qs = []

        with torch.no_grad():
            for i in range(0, all_states.shape[0], batch_size):
                batch_states = torch.tensor(all_states[i:i + batch_size].astype(np.float32), device=agent.device)
                q_vals = agent.policy_net(batch_states).cpu().numpy()
                Qs.append(q_vals)
        Qs = np.vstack(Qs)
        env.visualize_policy_execution(Qs, path=TRAINED_GIF_PATH)
        print(f"→ Trained policy GIF: {TRAINED_GIF_PATH}")

    if random_policy_gif_needed:
        RANDOM_GIF_PATH = os.path.join(gif_dir, f"{model_name}_random.gif")
        random_policy = np.ones_like(Qs)
        env.visualize_policy_execution(random_policy, path=RANDOM_GIF_PATH)
        print(f"→ Random policy GIF: {RANDOM_GIF_PATH}")

# Async Evaluation Thread
def run_async_eval_and_visualize(agent, ep, rewards_per_episode, plot_dir, model_dir):
    def task():
        try:
            print(f"\n[Async] Starting evaluation & visualization at episode {ep}...")
            eval_agent = DQNAgent(device='cpu')
            eval_agent.policy_net.load_state_dict(agent.policy_net.state_dict())
            eval_agent.target_net.load_state_dict(agent.target_net.state_dict())
            # ensure eval net is in eval mode (redundant but explicit)
            eval_agent.policy_net.eval()
            eval_agent.target_net.eval()

            eval_env = TreasureHunt_v2()
            evaluate(eval_agent, eval_env)
            visualiseModel(
                eval_agent, eval_env, list(rewards_per_episode),
                model_name=f"checkpoint_{ep}",
                gif_dir=GIF_DIR, plot_dir=plot_dir, model_dir=model_dir,
                random_policy_gif_needed=False
            )
            print(f"[Async] Done for episode {ep}.")
        except Exception as e:
            print(f"[Async ERROR] at episode {ep}: {e}")

    thread = threading.Thread(target=task, daemon=True)
    thread.start()
    return thread

# Training Loop
def train_dqn(env, agent, episodes=NUM_EPISODE_LEN, plot_dir=PLOT_DIR, model_dir=MODEL_DIR):
    rewards_per_episode, train_losses, ASYNC_JOBS = [], [], []

    for ep in tqdm(range(episodes), desc="Training Episodes"):
        s = env.reset()
        total_r = 0

        for _ in range(MAX_EPISODE_LENGTH):
            a = agent.act(s)
            ns, r = env.step(a)
            done = agent.is_terminal(ns)
            agent.replay_buffer.push(s, a, r, ns, done)
            loss = agent.update()
            if loss is not None:
                train_losses.append(loss)
            s = ns
            total_r += r
            if done:
                break

        agent.decay_epsilon()
        rewards_per_episode.append(total_r)

        if (ep+1) % 500 == 0:
            avg_r = np.mean(rewards_per_episode[-500:])
            max_r = np.max(rewards_per_episode[-500:])
            min_r = np.min(rewards_per_episode[-500:])
            print(f"Last (500ep), Ep {ep+1}: Avg={avg_r:.3f}, Max={max_r:.3f}, Min={min_r:.3f}, ε={agent.epsilon:.3f}")

        if (ep + 1) % 1000 == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
            print(f"[GPU CLEANUP] Cache cleared at episode {ep+1}")
            th = run_async_eval_and_visualize(agent, ep + 1, rewards_per_episode, plot_dir, model_dir)
            ASYNC_JOBS.append(th)

    visualiseModel(agent, env, rewards_per_episode,
                   model_name="trained treasurehunt",
                   gif_dir=GIF_DIR, plot_dir=plot_dir, model_dir=model_dir,
                   random_policy_gif_needed=True, is_time_stamp_needed = False)

    for th in ASYNC_JOBS:
        if th.is_alive():
            th.join()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("\nTraining complete.")
    plot_evaluation_curves(agent, eval_interval=1000)

# Main
if __name__ == "__main__":
    logging(s="q4")
    print("Starting DQN training on TreasureHunt_v2 environment")

    set_seed(SEED)
    env = TreasureHunt_v2()
    agent = DQNAgent()

    visualiseModel(agent, env, [],
                   model_name="INITIAL_MODEL",
                   gif_dir=GIF_DIR,
                   random_policy_gif_needed=True,
                   is_time_stamp_needed = False)

    start_time = time.time()
    warmup_replay_buffer(env, agent)
    train_dqn(env, agent, episodes=NUM_EPISODE_LEN)
    print(f"Training finished in {(time.time() - start_time) / 60:.2f} min")

    sys.stdout = original_stdout
    print("Logging stopped. Output restored to terminal.")
