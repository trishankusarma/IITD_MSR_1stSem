import os
import random
import gymnasium as gym
from datetime import datetime
import sys
import torch
import torch.nn as nn
import numpy as np
import copy
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from utils.loggerUtils import Logger
from utils.replayBuffer import ReplayBuffer
from utils.utils import generate_gif, plot_training_curves, store_evaluation_results, plot_q_values_per_action

# Hyper parameters
SEED = 42
NUM_EPISODES = 10000
MAX_NUM_STEPS = 1000
GAMMA = 0.99
LAYER_ARCHITECTURE = [128, 128]
LR = 5e-4
EPS_START = 1.00
EPS_END = 0.05
EPS_DECAY_RATE = (EPS_END/ EPS_START)**(1/NUM_EPISODES)
TARGET_UPDATE_FREQ = 2000
BUFFER_CAPACITY = 100000
MIN_BUFFER_SIZE = 5000
BUFFER_SIZE = 128
FINAL_EVALUATION_EPISODES = 100
UPDATE_FREQ = 4

# CONSTANTS
DQN = "DQN"
DOUBLE_DQN = "DOUBLE_DQN"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# NOW set global seed
set_seed(SEED)

# set up destiantions to store results
PLOTS_DIR = "plots"
MODELS_DIR = "models"
EVALUATION_DIR = "evaluation"
LOGS_DIR = "logs"
GIFS_DIR = "gifs"

os.makedirs(PLOTS_DIR, exist_ok = True)
os.makedirs(MODELS_DIR, exist_ok = True)
os.makedirs(EVALUATION_DIR, exist_ok = True)
os.makedirs(LOGS_DIR, exist_ok = True)
os.makedirs(GIFS_DIR, exist_ok = True)

# Neural Network 
class NonLinearModel(nn.Module):
    def __init__(self, input_dim = 8, output_dim = 4, hidden_layer1 = 256, hidden_layer2 = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_layer1),
            nn.ReLU(),
            nn.Linear(hidden_layer1, hidden_layer2),
            nn.ReLU(),
            nn.Linear(hidden_layer2, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)    

# Agent
class Agent():
    def __init__(self, env, device = device, gamma = GAMMA, layerArchitecture = LAYER_ARCHITECTURE, lr = LR, buffer_capacity = BUFFER_CAPACITY, batch_size = BUFFER_SIZE, algo = None):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.policy_net = NonLinearModel(input_dim = self.num_states, output_dim = self.num_actions, 
                                  hidden_layer1 = layerArchitecture[0], hidden_layer2 = layerArchitecture[1]
                                  ).to(device) 
        self.target_net = copy.deepcopy(self.policy_net).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.epsilon = EPS_START
        self.total_steps = 0
        self.replay_buffer = ReplayBuffer(
            capacity=int(buffer_capacity),
            state_shape=(self.num_states,),
            device=device
        )
        self.batch_size = batch_size
        self.algo = algo
    
    def select_action(self,state, greedy = False):
        if  not greedy and random.random() <= self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad(): 
            tensor_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) # get it back to dimension [1,8]
            q_value = self.policy_net(tensor_t) # make a forward pass through the target net
            return int(q_value.argmax(dim=1).item())
    
    # Warmup & Evaluation
    def warmup_replay_buffer(self, min_size=MIN_BUFFER_SIZE):
        print(f"[{self.algo}] Warming up replay buffer with {min_size} random transitions...")

        while len(self.replay_buffer) < min_size:
            state, _ = self.env.reset()  # gymnasium reset returns (obs, info)
            done = False

            for _ in range(MAX_NUM_STEPS):
                action = np.random.randint(self.num_actions)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store transition
                self.replay_buffer.push(state, action, reward, next_state, done)

                state = next_state

                if done:
                    break  # end episode
                if len(self.replay_buffer) >= min_size:
                    break  # stop once we have enough samples

        print(f"[{self.algo}] Replay buffer filled with {len(self.replay_buffer)} transitions.")

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            if self.algo == DQN:
                max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
                targets = rewards + self.gamma * (1 - dones) * max_next_q

            elif self.algo == DOUBLE_DQN:
                best_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, best_actions)
                targets = rewards + self.gamma * (1 - dones) * next_q

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def train(self, seed):
        set_seed(seed, self.env)  # Seed both global RNG and environment
        train_losses = []
        episode_rewards = []

        # tracking best model
        best_avg_reward = -float('inf')
        best_model_state = None
        best_episode = 0

        for ep in tqdm(range(NUM_EPISODES), desc="Training Episodes"):
            state, _ = self.env.reset()
            episode_reward = 0
            
            for step in range(MAX_NUM_STEPS):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done_flag = terminated or truncated
                
                self.total_steps += 1
                episode_reward += reward
                
                self.replay_buffer.push(state, action, reward, next_state, done_flag)

                if step % UPDATE_FREQ == 0 or done_flag:
                    loss = self.update()
                    if loss is not None:
                        train_losses.append(loss)
                
                state = next_state
                
                if self.total_steps % TARGET_UPDATE_FREQ == 0: # load the target net every 2000 steps
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                if done_flag:
                    break
    
            episode_rewards.append(episode_reward)
            if (ep + 1) % 100 == 0:
                recent_rewards = episode_rewards[-100:]
                avg_reward = np.mean(recent_rewards)

                # Save if this is the best so far
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_model_state = copy.deepcopy(self.policy_net.state_dict())
                    best_episode = ep + 1
                    print(
                        f"[{self.algo}] Episode {ep+1}/{NUM_EPISODES}, "
                        f"Epsilon={self.epsilon:.3f}, "
                        f"Mean(100)={avg_reward:.2f}, "
                        f"Std(100)={np.std(recent_rewards):.2f}, "
                        f"Max(100)={np.max(recent_rewards):.2f} ← NEW BEST!"
                    )
                else:
                    print(
                        f"[{self.algo}] Episode {ep+1}/{NUM_EPISODES}, "
                        f"Epsilon={self.epsilon:.3f}, "
                        f"Mean(100)={avg_reward:.2f}, "
                        f"Std(100)={np.std(recent_rewards):.2f}, "
                        f"Max(100)={np.max(recent_rewards):.2f}"
                    )
            self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY_RATE)

        # Load the best model before saving
        if best_model_state is not None:
            self.policy_net.load_state_dict(best_model_state)
            print(f"[{self.algo}] Loaded best model from episode {best_episode} (avg reward: {best_avg_reward:.2f})")
        
        model_name = "dqn.pt" if self.algo == DQN else "ddqn.pt"
        torch.save(self.policy_net.state_dict(), f"models/{model_name}")
        print(f"[{self.algo}] Model saved to models/{model_name}")
        return episode_rewards, train_losses
                
    def evaluate(self, env_eval, num_episodes=100, greedy=True):
        old_eps = self.epsilon
        self.epsilon = 0.0 if greedy else self.epsilon
        self.policy_net.eval()
        rewards = []
        q_values_all = []

        for idx in tqdm(range(num_episodes), desc="Evaluating Episodes"):
            state, _ = env_eval.reset(seed=SEED + idx)
            done = False
            total_reward = 0
            step = 0

            while not done and step < MAX_NUM_STEPS:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
                q_values_all.append(q_values)

                action = np.argmax(q_values)
                next_state, reward, terminated, truncated, _ = env_eval.step(action)
                done = terminated or truncated
                total_reward += reward
                state = next_state
                step += 1

            rewards.append(total_reward)

        self.policy_net.train()
        self.epsilon = old_eps

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        maxm_reward = float(np.max(rewards))
        minm_reward = float(np.min(rewards))

        print(f"[{self.algo}] Evaluation → mean: {mean_reward:.2f}, std: {std_reward:.2f}, max: {maxm_reward:.2f}, min: {minm_reward:.2f}")

        return np.array(q_values_all), {
            "mean_return": mean_reward,
            "std_return": std_reward,
            "max_reward": maxm_reward,
            "min_reward": minm_reward,
            "all_episode_rewards": rewards
        }
        
if __name__ == '__main__':  
    logging(s="Q1")
    print("Starting with assignment 4 :: Q1 :: Env : LunarLander-v3")
    
    # Question 1 : Implementation
    print("Running with DQN algo")
    env1 = gym.make('LunarLander-v3')
    env_eval1 = gym.make('LunarLander-v3', render_mode="rgb_array")
    agent_dqn = Agent(env1, algo = DQN)

    for name, param in agent_dqn.policy_net.named_parameters():
        print("DQN :: Policy Net :: ",name, param.shape)

    for name, param in agent_dqn.target_net.named_parameters():
        print("DQN :: Target Net :: ",name, param.shape)

    agent_dqn.warmup_replay_buffer()
    episode_rewards_dqn, train_losses_dqn = agent_dqn.train(SEED) # worked on single seed
    q_values_all_dqn, eval_stats_dqn = agent_dqn.evaluate(env_eval1, num_episodes = FINAL_EVALUATION_EPISODES)
    
    print("Running with Double DQN algo")

    env2 = gym.make('LunarLander-v3')
    env_eval2 = gym.make('LunarLander-v3', render_mode="rgb_array")

    agent_ddqn = Agent(env2, algo = DOUBLE_DQN)

    for name, param in agent_ddqn.policy_net.named_parameters():
        print("Double DQN :: Policy Net :: ",name, param.shape)

    for name, param in agent_ddqn.target_net.named_parameters():
        print("Double DQN :: Target Net :: ",name, param.shape)

    agent_ddqn.warmup_replay_buffer()
    episode_rewards_ddqn, train_losses_ddqn = agent_ddqn.train(SEED) # worked on single seed
    q_values_all_ddqn, eval_stats_ddqn = agent_ddqn.evaluate(env_eval2, num_episodes = FINAL_EVALUATION_EPISODES)
    
    # Question 2: Save plots
    plot_training_curves(
        dqn_rewards=episode_rewards_dqn,
        ddqn_rewards=episode_rewards_ddqn,
        dqn_losses=train_losses_dqn,
        ddqn_losses=train_losses_ddqn,
        window_size=50,
        save_dir=PLOTS_DIR
    )
    
    # Question 3 :: storing the evaluation results we got
    store_evaluation_results(dqn_eval = eval_stats_dqn, ddqn_eval=eval_stats_ddqn, save_dir = "evaluation", save_file_name = "evaluation_results.json")
    
    # Question 4: 
    plot_q_values_per_action(
        dqn_qvalues=q_values_all_dqn, ddqn_qvalues=q_values_all_ddqn, save_dir=PLOTS_DIR, smooth_window=20
    )

    # Generate GIF
    generate_gif(env_eval1, agent_dqn, f"{GIFS_DIR}/dqn_final.gif")
    generate_gif(env_eval2, agent_ddqn, f"{GIFS_DIR}/ddqn_final.gif")

    env1.close()
    env2.close()
    env_eval1.close()
    env_eval2.close()