import gymnasium
import gym_gridworlds
import numpy as np
import random
import imageio
import os
from collections import defaultdict
from gymnasium.envs.registration import register
from gym_gridworlds.behaviour_policies import create_behaviour
import json
from .utils.plotUtils import plot  

GRID_ROWS = 4
GRID_COLS = 5

# Register the custom environment
register(
    id="Gym-Gridworlds/Full-4x5-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=500,
    kwargs={"grid": "4x5_full"},
)

def getDecayedEpsilon(ep, num_episodes, start_epsilon=1.0, end_epsilon=0.05):
    # Compute linear decay
    slope = (end_epsilon - start_epsilon) / num_episodes
    current_epsilon = max(end_epsilon, start_epsilon + slope * ep)
    return current_epsilon

# def getDecayedEpsilon(ep, num_episodes, start_epsilon=1.0, end_epsilon=0.05, decay_rate=0.9993):
#     currentEpsilon = start_epsilon * (decay_rate ** episode)
#     return max(end_epsilon, currentEpsilon)

# Linear epsilon decay
def getDecayedAlpha(episode, num_episodes, start_alpha=0.1, end_alpha=0.05):
    decay_rate = (start_alpha - end_alpha) / num_episodes
    return max(end_alpha, start_alpha - decay_rate * episode)

def set_global_seed(seed: int):
    """Set seed for reproducibility across modules."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def state_to_cord(state):
    return divmod(state, GRID_COLS)

def cord_to_state(row, col):
    return row * GRID_COLS + col

def reset_fn(env, seed=10):
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env.np_random, _ = gymnasium.utils.seeding.np_random(seed)
    
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

def tdis(env, num_episodes=5000, seed=0, gamma=0.99, noise=0.0):
    """
    Off-policy TD(0) with per-decision importance sampling.
    """
    num_actions = env.action_space.n
    set_global_seed(seed)
    reset_fn(env, seed=seed)

    Q = defaultdict(lambda: np.zeros(num_actions, dtype=float))
    behavior_Q = create_behaviour(noise)
    
    def target_policy(state):
        return np.argmax(Q[state])
    
    def behavior_policy(state):
        probs = behavior_Q[state]
        return np.random.choice(len(probs), p=probs)

    def get_behavior_prob(state, action):
        return behavior_Q[state][action]

    def get_target_prob(state, action, epsilon=0.5):
        action_values = Q[state]
        greedy_actions = np.flatnonzero(action_values == np.max(action_values))
        probs = np.ones(num_actions) * (epsilon / num_actions)
        probs[greedy_actions] += (1 - epsilon) / len(greedy_actions)
        return probs[action]
    
    def get_target_prob_distribution(state, epsilon=0.5):
        action_values = Q[state]
        greedy_actions = np.flatnonzero(action_values == np.max(action_values))
        probs = np.ones(num_actions) * (epsilon / num_actions)
        probs[greedy_actions] += (1 - epsilon) / len(greedy_actions)
        return probs

    episode_rewards = []
    max_steps = getattr(env.spec, "max_episode_steps", 500)

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0.0
        epsilon = getDecayedEpsilon(ep, num_episodes)
        
        alpha = getDecayedAlpha(ep, num_episodes) # make this vairable from 0.1 to 0.05

        while not done and steps < max_steps:
            action = behavior_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            pi_prob = get_target_prob(state, action, epsilon=epsilon)
            b_prob = get_behavior_prob(state, action)

            rho = 0.0 if b_prob == 0 else pi_prob / b_prob

            target_prob_distribution = get_target_prob_distribution(next_state, epsilon=epsilon)
            expected_prob_distribution = np.dot(Q[next_state], target_prob_distribution)
            td_target = reward + (gamma * expected_prob_distribution if not done else 0.0)
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * rho * td_error

            state = next_state
            steps += 1

        episode_rewards.append(total_reward)

    num_states = env.observation_space.n
    final_policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        final_policy[s] = int(np.argmax(Q[s]))

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
            action = policy[state]
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            state = next_state
            steps += 1

        rewards.append(episode_reward)
        if episode_reward > 0.5:
            success_rate += 1

    return (np.mean(rewards), np.min(rewards), np.max(rewards),
            np.std(rewards), success_rate / n_episodes)

def generate_policy_gif(env, policy, gif_filename, max_steps=500):
    frames = []
    print(f"\nGenerating GIF... saving to {gif_filename}")
    env_render = gymnasium.make(env.spec.id, render_mode='rgb_array', random_action_prob=0.1)
    reset_fn(env_render, 504)
    state, _ = env_render.reset(seed=504)
    done = False
    steps = 0
    total_rewards = 0

    while not done and steps < max_steps:
        action = policy[state]
        frames.append(env_render.render())
        state, reward, terminated, truncated, _ = env_render.step(action)
        done = terminated or truncated
        steps += 1
        total_rewards += reward

    env_render.close()
    imageio.mimsave(gif_filename, frames, fps=7)
    print(f"✅ GIF saved as {gif_filename}")
    print(f"Total reward obtained : {total_rewards}")

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    num_seeds = 10
    num_episodes = 5000
    gamma = 0.99

    env = gymnasium.make('Gym-Gridworlds/Full-4x5-v0', random_action_prob=0.1)
    print(f"--- Starting training across {num_seeds} seeds ---")

    noises = [0.0, 0.1, 0.01]

    for noise in noises:
        all_rewards = []

        best_policy = None
        best_q_values = None
        best_mean = -np.inf
        
        print(f"Training for noise : {noise}")

        for seed in range(num_seeds):
            print(f"\n--- Training Seed {seed + 1}/{num_seeds} ---")

            q_values, policy, rewards = tdis(env, num_episodes=num_episodes, seed=seed,
                                                   gamma=gamma, noise=noise)
            all_rewards.append(rewards)

            mean_reward, min_reward, max_reward, std_reward, success_rate = evaluate_policy(env, policy)
            print(f"\nSeed {seed}: Mean={mean_reward:.3f}, Min={min_reward:.3f}, "
                  f"Max={max_reward:.3f}, Std={std_reward:.3f}, Success Rate={success_rate:.3f}")

            if mean_reward > best_mean:
                best_mean = mean_reward
                best_policy = policy.copy()
                best_q_values = q_values.copy()

        # Truncate rewards to shortest length across seeds for averaging
        min_len = min(map(len, all_rewards))
        truncated_rewards = np.array([r[:min_len] for r in all_rewards])
        mean_rewards = np.mean(truncated_rewards, axis=0)
        std_rewards = np.std(truncated_rewards, axis=0)

        path = f"gym_gridworlds/plots/temporal_difference_reward_curve_{noise}.png"
        plot(data=mean_rewards, min_len=min_len, mean=mean_rewards, std=std_rewards,
             label='Mean reward', xlabel='Episode', ylabel='Reward',
             title='TD(0) Off-Policy Mean Learning Curve (All Seeds)',
             filename=path)
        print("✅ Saved mean learning curve.")

        mean_reward, min_reward, max_reward, std_reward, success_rate = evaluate_policy(env, best_policy)
        key = f"TD0_ImportanceSampling({noise})"
        value = (mean_reward, std_reward)
        json_path = 'gym_gridworlds/evaluation/importance_sampling_evaluation_results.json'
        json_update(json_path, key, value)
        print(f"Noise : {noise} :: {key}: {value}")

        action_map = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up', 4: 'Stay'}
        
        # Optional: uncomment to generate GIF
        print("\nBest Optimal Policy (Action to take in each state):")
        if best_policy is not None:
            policy_grid = np.array([action_map[i] for i in best_policy]).reshape(GRID_ROWS, GRID_COLS)
            print(policy_grid)

            print("\nQ-values for State 0 (top-left) from the best policy:")
            if 0 in best_q_values:
                for action, value in enumerate(best_q_values[0]):
                    print(f"  Action: {action_map[action]}, Q-value: {value:.3f}")
                    
            generate_policy_gif(env, best_policy, f"gym_gridworlds/gifs/temporal_difference_{noise}.gif", max_steps=50)
    env.close()
