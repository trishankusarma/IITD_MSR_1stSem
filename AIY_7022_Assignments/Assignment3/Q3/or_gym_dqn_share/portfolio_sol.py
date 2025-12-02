import sys
import gymnasium as gym
sys.modules["gym"] = gym  

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.envs.registration")

from utils.config import *
from utils.agent import DQNAgent
from utils.train import train_loop
from utils.evaluate import evaluate
from utils.actions import ALL_ACTIONS
from utils.loggerUtils import Logger
from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv
import torch, matplotlib.pyplot as plt, time
from datetime import datetime

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

# HyperParameters
NUM_EPISODES = 35000

MODEL_DIR = "models"
import os
os.makedirs(MODEL_DIR, exist_ok = True)

if __name__ == "__main__":
    logging(s="q3")
    set_seed(SEED)
    num_actions = len(ALL_ACTIONS)
    start = time.time()

    # # Task 1 — Terminal Wealth
    # print("\n=== TRAINING: TASK A ===")
    # env1 = DiscretePortfolioOptEnv(env_config=get_env_config(SEED))
    # obs = env1.reset()
    # agent1 = DQNAgent(
    #     n_obs=len(obs),
    #     num_actions=len(ALL_ACTIONS),
    #     device=DEVICE,
    #     buffer_size=50_000,
    #     batch_size=128,
    #     lr=1e-4,
    #     gamma=0.99
    # )
    # agent1.warmup_replay_buffer(env1)
    # returns1, losses1 = train_loop(env1, agent1, "a", num_episodes = NUM_EPISODES)
    # torch.save(agent1.policy_net.state_dict(), MODEL_PATH_TASK1)
    # print(f"Saved Task 1 → {MODEL_PATH_TASK1}")
    # evaluate(agent1, lambda seed: DiscretePortfolioOptEnv(env_config=get_env_config(seed)),
    #          num_seeds=100, max_steps=10, title="Task A Terminal Wealth", 
    #          save_dir="evaluationForPartA", training_rewards=returns1, training_losses=losses1)

    # Task 2 — Step Wealth
    print("\n=== TRAINING: TASK B ===")
    env2 = DiscretePortfolioOptEnv(env_config=get_env_config(SEED))
    obs = env2.reset()
    agent2 = DQNAgent(
        n_obs=len(obs),
        num_actions=len(ALL_ACTIONS),
        device=DEVICE,
        buffer_size=50_000,
        batch_size=128,
        lr=1e-4,
        gamma=0.99
    )
    returns2, losses2 = train_loop(env2, agent2, "b", num_episodes = NUM_EPISODES)
    torch.save(agent2.policy_net.state_dict(), MODEL_PATH_TASK2)
    print(f"Saved Task 2 → {MODEL_PATH_TASK2}")
    evaluate(agent2, lambda seed: DiscretePortfolioOptEnv(env_config=get_env_config(seed)),
             num_seeds=100, max_steps=10, title="Task B Step Wealth", save_dir="evaluationForPartB", 
             training_rewards=returns2, training_losses=losses2)

    print(f"\nTraining Done in {time.time()-start:.2f}s")
