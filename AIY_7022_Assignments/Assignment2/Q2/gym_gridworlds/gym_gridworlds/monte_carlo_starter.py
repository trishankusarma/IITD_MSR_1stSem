import gymnasium
import gym_gridworlds
import numpy as np
import random
import imageio
import matplotlib.pyplot as plt 
from collections import defaultdict
from gymnasium.envs.registration import register
from gym_gridworlds.behaviour_policies import create_behaviour
from .utils.plotUtils import plot
import json
import os

GRID_ROWS = 4
GRID_COLS = 5

# Register the custom environment
register(
    id="Gym-Gridworlds/Full-4x5-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=500,
    kwargs={"grid": "4x5_full"},
)

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def reset_fn(env, seed=10):
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env.np_random, _ = gymnasium.utils.seeding.np_random(seed)

def state_to_cord(state):
    return divmod(state, GRID_COLS)

def cord_to_state(row, col):
    return row * GRID_COLS + col

# 0.9991 -- this decay rate worked for 0.01 noise 
# worked for 0.1 also
def getExponentialDecayedEpsilon(episode, start_epsilon=1.0, end_epsilon=0.05, decay_rate=0.9991):
    currentEpsilon = start_epsilon * (decay_rate ** episode)
    return max(end_epsilon, currentEpsilon)

def json_update(filename, key, value):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    if key not in data:
        data[key] = {}
    data[key]['mean'] = value[0]
    data[key]['std'] = value[1]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON file '{filename}' updated successfully.")

def monte_carlo_off_policy_control(env, num_episodes=5000, seed=0, gamma=0.9, noise=0.0):
    """
    Monte Carlo Off-Policy Control with Weighted Importance Sampling.
    Behavior policy: stochastic (given by behavior_Q)
    Target policy: deterministic greedy wrt Q
    """
    num_actions = env.action_space.n
    num_states = env.observation_space.n

    Q = defaultdict(lambda: np.ones(num_actions))
    C = defaultdict(lambda: np.zeros(num_actions))  # cumulative weights

    set_global_seed(seed)
    reset_fn(env, seed=seed)
    
    # Create behavior policy with noise parameter
    behavior_Q = create_behaviour(noise)

    def target_policy_action(state):
        """Greedy action wrt Q."""
        return np.argmax(Q[state])

    def behavior_policy_action(state):
        probs = behavior_Q[state]
        return np.random.choice(len(probs), p=probs)

    def get_behavior_prob(state, action):
        return behavior_Q[state][action]

    def get_target_prob(state, action, epsilon=0.1):  # pure greedy target policy
        greedy_action = np.argmax(Q[state])
        if action == greedy_action:
            return 1 - epsilon + epsilon / num_actions
        else:
            return epsilon / num_actions
        
    episode_rewards = []
    for ep in range(num_episodes):
        reset_fn(env, seed)
        episode = []  # list of (s, a, r)
        s, _ = env.reset(seed=seed)
        current_epsilon = getExponentialDecayedEpsilon(ep)
        done = False
        total_rewards = 0
        
        while not done:
            a = behavior_policy_action(s)
            s_next, reward, terminated, truncated, _ = env.step(a)
            episode.append((int(s), int(a), reward))
            done = terminated or truncated
            s = s_next
            total_rewards += reward
        
        episode_rewards.append(reward)

        # Compute returns backwards and update with weighted importance sampling
        G = 0.0
        W = 1.0

        for (s, a, r) in reversed(episode):
            G = gamma * G + r
            C[s][a] += W
            Q[s][a] += (W / C[s][a]) * (G - Q[s][a])

            b_prob = get_behavior_prob(s, a)
            pi_prob = get_target_prob(s, a, epsilon = current_epsilon)
            
            rho = 0.0 if b_prob == 0 else pi_prob / b_prob
            
            W = W * rho
            if W == 0:
                break  # stop updating if importance weight becomes zero

    final_policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        final_policy[s] = np.argmax(Q[s])

    return Q, final_policy, episode_rewards

def evaluate_policy(env, policy, n_episodes=100, max_steps=500):
    rewards = []
    success_rate = 0

    for episode in range(n_episodes):
        state, _ = env.reset(seed=episode)
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < max_steps:
            action = int(policy[state])
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
            
        rewards.append(episode_reward)
        if episode_reward > 0.5:
            success_rate += 1

    return (np.mean(rewards), np.min(rewards), np.max(rewards),
            np.std(rewards), success_rate / n_episodes)

def generate_policy_gif(env, policy, filename='policy_run_mcmc.gif', max_steps=500):
    frames = []
    print(f"\nGenerating GIF... saving to {filename}")

    env_render = gymnasium.make(env.spec.id, render_mode='rgb_array', random_action_prob=0.1)
    reset_fn(env_render, 504)
    state, _ = env_render.reset(seed=504)
    done = False
    steps = 0
    total_reward = 0

    frames.append(env_render.render())

    while not done and steps < max_steps:
        action = int(policy[state])
        next_state, reward, terminated, truncated, _ = env_render.step(action)
        frames.append(env_render.render())
        done = terminated or truncated
        state = next_state
        steps += 1
        total_reward += reward

    env_render.close()
    imageio.mimsave(filename, frames, fps=7)
    print(f"Reward : {reward}")
    print(f"GIF saved successfully to {os.path.abspath(filename)}")

if __name__ == '__main__':
    num_seeds = 10
    gamma = 0.9
    noise_values = [0.0, 0.1, 0.01]
    num_episodes = [8000, 5000, 5000]

    env = gymnasium.make('Gym-Gridworlds/Full-4x5-v0', random_action_prob=0.1)
    
    for index,noise in enumerate( noise_values ):
        best_policy = None
        best_q_values = None
        best_mean = -np.inf

        print(f"--- Starting training across {num_seeds} seeds ---Running for {num_episodes[index]} episodes and Noise : {noise} --")

        all_rewards = []
        for seed in range(num_seeds):
            print(f"\n--- Training Seed {seed + 1}/{num_seeds} ---")
                
            # Train the policy
            q_values, policy, episode_rewards = monte_carlo_off_policy_control(env, num_episodes[index], seed, gamma, noise)
            all_rewards.append(episode_rewards)
                
            # Evaluate the train policy
            mean_reward, min_reward, max_reward, std_reward, success_rate = evaluate_policy(env, policy)
            print(f"\nSeed {seed}: Mean={mean_reward:.3f}, Min={min_reward:.3f}, Max={max_reward:.3f}, "
                  f"Std={std_reward:.3f}, Success Rate={success_rate:.3f}")

            if mean_reward > best_mean:
                best_mean = mean_reward
                best_policy = policy.copy() # Question 2:: Training phase :: returning the best learned Q table for monte-carlo
                best_q_values = q_values.copy()
            
        # Question 3 :: Evaluation phase
        mean_reward, min_reward, max_reward, std_reward, success_rate = evaluate_policy(env, best_policy)
        key = f"MC_ImportanceSampling({noise})"
        value = (mean_reward, std_reward)
        json_path = 'gym_gridworlds/evaluation/importance_sampling_evaluation_results.json'
        json_update(json_path, key, value)
        print(f"Noise : {noise} :: {key}: {value}")
        
            
        # Truncate rewards to shortest length across seeds for averaging
        # Question 4: Learning Analysis
        min_len = min(map(len, all_rewards))
        truncated_rewards = np.array([r[:min_len] for r in all_rewards])
        mean_rewards = np.mean(truncated_rewards, axis=0)
        std_rewards = np.std(truncated_rewards, axis=0)

        path = f"gym_gridworlds/plots/monte_carlo_reward_curve_{noise}.png"
        plot(data=mean_rewards, min_len=min_len, mean=mean_rewards, std=std_rewards,
                 label='Mean reward', xlabel='Episode', ylabel='Reward',
                 title='Monte Carlo Weighted Off-Policy Mean Learning Curve (All Seeds)',
                 filename=path)
        print("✅ Saved mean learning curve.")
            
        # Question 5: Policy demonstration
        print("\nBest Optimal Policy (Action to take in each state):")
        if best_policy is not None:
            action_map = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'Stay'}
            policy_grid = np.array([action_map[i] for i in best_policy]).reshape(GRID_ROWS, GRID_COLS)
            print("\nBest Optimal Policy (Action per state):")
            print(policy_grid)
                
            print("\nQ-values for State 0 (top-left) from the best policy:")
            if 0 in best_q_values:
                for action, value in enumerate(best_q_values[0]):
                    print(f"  Action: {action_map[action]}, Q-value: {value:.3f}")
                
                generate_policy_gif(env, best_policy, filename=f"gym_gridworlds/gifs/monte_carlo_noise{noise}.gif")
        
    env.close()
