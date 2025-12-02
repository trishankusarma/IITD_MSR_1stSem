# Code for Part b
import gymnasium as gym
import torch
import os
from q2_run_training_policies import set_seed, logging, generate_episode
import json
from tqdm import tqdm
from reinforce_baselines import PolicyNetwork

SEED = 42
HIDDEN_DIM = 64
ENV_NAME = "InvertedPendulum-v4"

os.makedirs("trajectories", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_policy(path, env):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]

    policy = PolicyNetwork(input_dim, output_dim, hidden_dim = HIDDEN_DIM).to(device)

    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["policy"]

    policy.load_state_dict(state_dict)
    policy.eval()

    return policy

def collect_n_trajectories(baseline_type, num_episodes = 500):

    # 1. create env
    env = gym.make(ENV_NAME)
    env.reset(seed=SEED)
    set_seed(SEED, env)  # Seed both global RNG and environment

    action_low = env.action_space.low
    action_high = env.action_space.high

    # 2. a) Load the respective models to the baseline
    model_path = f"models/reinforce_{baseline_type}.pt"
    policy = load_policy(model_path, env)

    trajectories = []
    
    # 3. generate 500 episodes of each env
    for ep in tqdm(range(num_episodes), desc=f"Collecting episodes : {baseline_type}"):
        states, actions, rewards, log_probs = generate_episode(env, policy, action_low, action_high)

        log_probs = [float(lp.cpu().item()) for lp in log_probs]

        trajectories.append({
            "states": [s.tolist() for s in states],
            "actions": [a.tolist() if hasattr(a, "tolist") else a for a in actions],
            "rewards": rewards,
            "log_probs": log_probs
        })
    
    trajectory_path = f"trajectories/reinforce_{baseline_type}.json"
    with open(trajectory_path, "w") as f:
        json.dump(trajectories, f, indent=4)

if __name__ == "__main__":
    logging(s="Q2_2")
    baselines = [
        "no_baseline",
        "reward_to_go",
        "avg_reward",
        "value_function"
    ]
    
    for baseline in baselines:
        collect_n_trajectories(baseline, num_episodes = 500)