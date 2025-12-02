import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import os
import json
import matplotlib.pyplot as plt
from cliff import MultiGoalCliffWalkingEnv
from utils.replayBufferClass import PrioritizedReplayBuffer
from datetime import datetime
import sys
from tqdm import tqdm 
from utils.loggerUtils import Logger

# GPU support added here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Hyperparameters
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_SIZE = 100000
TARGET_UPDATE = 100
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_LINEAR = 0.99992
EPS_DECAY_NONLINEAR = 0.9999
LR_LINEAR = 1e-2
LR_NONLINEAR = 1e-3
NUM_EPISODES = 25000
EVAL_EPISODES = 100
SEED = 42

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('evaluation', exist_ok=True)

#  Helpers 
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


def process_state(state, env):
    """
      One-hot encode full state index (env.observation_space.n)
      Move tensor to GPU automatically
    """
    vec = np.zeros(env.observation_space.n, dtype=np.float32)
    vec[state] = 1.0
    return torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)

def smooth_rewards(rewards, window=100):
    smoothed = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(rewards[start:i+1]))
    return smoothed

#  DQN Networks 
class LinearDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class NonLinearDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

#  DQN Trainer 
class DQNTrainer:
    def __init__(self, env, network, optimizer, eps_decay, seed=SEED):
        self.env = env
        self.net = network.to(device)         # Move model to GPU
        self.target_net = copy.deepcopy(network).to(device)  # GPU
        self.optimizer = optimizer
        self.memory = PrioritizedReplayBuffer(capacity=REPLAY_SIZE, state_shape=(env.observation_space.n,))
        self.epsilon = EPS_START
        self.gamma = GAMMA
        self.eps_decay = eps_decay
        set_seed(seed)
        self.targetSteps = 0

    def select_action(self, state_tensor):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.net(state_tensor).argmax(dim=1).item()
        else:
            return self.env.action_space.sample()

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # Sample a batch of transitions
        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(BATCH_SIZE)

        # Move batch tensors to GPU (they're already on GPU from the buffer)
        # No need to call .to(device) again if buffer is already on GPU
        
        # Squeeze to ensure correct shapes
        actions = actions.squeeze(1)      # (batch_size, 1) -> (batch_size,)
        rewards = rewards.squeeze(1)      # (batch_size, 1) -> (batch_size,)
        dones = dones.squeeze(1)          # (batch_size, 1) -> (batch_size,)
        weights = weights.view(-1)        # Ensure it's (batch_size,)

        # Compute Q values for the current states
        q_values = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch_size,)

        # Compute target Q values for the next states
        with torch.no_grad():
            next_actions = self.net(next_states).argmax(dim=1)  # (batch_size,)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)  # (batch_size,)
            target = rewards + (1 - dones) * self.gamma * next_q_values  # (batch_size,)

        # Compute the TD error
        td_errors = target - q_values  # (batch_size,)

        # Use importance sampling weights for loss
        loss = (weights * td_errors ** 2).mean()  # Weighted loss

        # Backpropagate and optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.optimizer.step()

        # Update the priorities of the sampled batch
        self.memory.update_priorities(indices, td_errors)


    def train(self, num_episodes=NUM_EPISODES, target_update=TARGET_UPDATE):
        rewards_per_episode = []
        best_avg_reward = -float('inf')
        best_state_dict = None

        for episode in tqdm(range(num_episodes), desc="Training Episodes"):
            state, _ = self.env.reset()
            state_tensor = process_state(state, self.env)
            total_reward = 0
            done = False
            step = 0

            while not done and step < 1000:
                action = self.select_action(state_tensor)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state_tensor = process_state(next_state, self.env)

                self.memory.push(state_tensor, action, reward, next_state_tensor, terminated)

                state_tensor = next_state_tensor
                total_reward += reward
                done = terminated or truncated
                step += 1
                self.targetSteps += 1

                self.optimize_model()

            rewards_per_episode.append(total_reward)

            if episode % 500 == 0:
                print(f"Episode {episode}: Reward = {total_reward:.2f}, Epsilon = {self.epsilon:.3f}")

            if len(rewards_per_episode) >= 50:
                avg_last50 = np.mean(rewards_per_episode[-50:])
                if avg_last50 > best_avg_reward:
                    best_avg_reward = avg_last50
                    best_state_dict = copy.deepcopy(self.net.state_dict())

            self.epsilon = max(EPS_END, self.epsilon * self.eps_decay)
            if self.targetSteps % target_update == 0:
                self.target_net.load_state_dict(self.net.state_dict())

        return rewards_per_episode, best_state_dict

    def evaluate(self, num_episodes=EVAL_EPISODES):
        old_eps = self.epsilon
        self.epsilon = 0.0
        rewards = []

        for ep in range(num_episodes):
            state, _ = self.env.reset()
            state_tensor = process_state(state, self.env)
            total_reward = 0
            done = False
            step = 0

            while not done and step < 1000:
                with torch.no_grad():
                    action = self.net(state_tensor).argmax(dim=1).item()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                state_tensor = process_state(next_state, self.env)
                total_reward += reward
                done = terminated or truncated
                step += 1

            rewards.append(total_reward)

        self.epsilon = old_eps
        return np.mean(rewards), np.std(rewards)

#  Training & Evaluation 
def train_single_model(model_class, model_name, env_train, lr, eps_decay):
    print(f"Training {model_name} model...")
    input_dim = env_train.observation_space.n
    output_dim = env_train.action_space.n

    model = model_class(input_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = DQNTrainer(env_train, model, optimizer, eps_decay)

    rewards, best_state_dict = trainer.train()
    torch.save(best_state_dict, f'models/best_{model_name}.pt')

    plt.figure(figsize=(10, 4))
    plt.plot(smooth_rewards(rewards))
    plt.title(f"{model_name.capitalize()} DQN Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.savefig(f'plots/cliff_average_rewards_{model_name}.png')
    plt.close()

    return rewards

def evaluate_model(model_class, model_path, env_eval, model_name):
    print(f"Evaluating model {model_name} from {model_path}")
    input_dim = env_eval.observation_space.n
    output_dim = env_eval.action_space.n

    model = model_class(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=LR_LINEAR)
    trainer = DQNTrainer(env_eval, model, optimizer, eps_decay=EPS_DECAY_NONLINEAR)
    mean_r, std_r = trainer.evaluate()

    print(f"{model_name} Evaluation - Mean Reward: {mean_r:.2f}, Std: {std_r:.2f}")
    return mean_r, std_r

def save_evaluation_results(results):
    with open('evaluation/cliff_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved evaluation results.")

#  Main 
def main():
    logging(s="q1")
    env_train = MultiGoalCliffWalkingEnv(train=True)
    env_eval = MultiGoalCliffWalkingEnv(train=False)

    train_single_model(LinearDQN, 'linear', env_train, lr=LR_LINEAR, eps_decay=EPS_DECAY_LINEAR)
    train_single_model(NonLinearDQN, 'nonlinear', env_train, lr=LR_NONLINEAR, eps_decay=EPS_DECAY_NONLINEAR)

    evaluation_results = {}
    for model_name, model_class in [('linear', LinearDQN), ('nonlinear', NonLinearDQN)]:
        model_path = f'models/best_{model_name}.pt'
        mean_r, std_r = evaluate_model(model_class, model_path, env_eval, model_name)
        evaluation_results[model_name] = {'mean': mean_r, 'std': std_r}

    save_evaluation_results(evaluation_results)
    env_train.close()
    env_eval.close()

if __name__ == "__main__":
    set_seed(SEED)
    main()
