import numpy as np
import os
import matplotlib.pyplot as plt
from cliff import set_global_seed
from utils.plotUtils import plot_graph

def get_decayed_epsilon(episode, decay_rate, min_epsilon, extreme, found_success, start_epsilon=1.0):
    if extreme:
        if not found_success:
            return 0.99
    return max(min_epsilon, start_epsilon * (decay_rate ** episode))

def get_decayed_alpha(episode, start_alpha=0.7, decay_rate=0.99997, min_alpha=0.5):
    return max(min_alpha, start_alpha * (decay_rate ** episode))

def adaptive_alpha(episode, recent_success_count, start_alpha=0.2, max_alpha=0.8, min_alpha=0.4,
                   success_threshold=10, alpha_decay_rate=0.9995):
    ramp = 1 / (1 + np.exp(-(recent_success_count - success_threshold) / (success_threshold + 1e-5)))
    alpha_now = start_alpha + (max_alpha - start_alpha) * ramp
    alpha_now = max(min_alpha, alpha_now * (alpha_decay_rate ** episode))
    return alpha_now

def extreme_alpha(ep, found_success, episode_success, start_alpha=0.2, max_alpha=0.8, min_alpha=0.4, alpha_decay_rate=0.9995):
    if not found_success:
        return start_alpha
    if episode_success:
        return max(min_alpha, max_alpha * (alpha_decay_rate ** ep))
    return start_alpha

def epsilon_greedy(Q, state, episode, decay_rate, min_epsilon, extreme, found_success):
    eps = get_decayed_epsilon(episode, decay_rate, min_epsilon, extreme, found_success)
    if np.random.rand() < eps:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[int(state)])

def run_monte_carlo_frozen_lake(
    env,
    training_seed=42,
    num_episodes=100000,
    gamma=0.999,
    max_steps=100,
    filename="monteCarlo.png",
    epsilon_decay_rate=0.99995,
    min_epsilon=0.2,
    success_threshold=10,
    start_alpha=0.3,
    max_alpha=0.8,
    min_alpha=0.2,
    alpha_decay_rate=0.999,
    extreme=True,
    plot_window=3
):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    print(f"Number of states: {num_states}")
    print(f"Number of actions: {num_actions}")

    all_rewards = []
    set_global_seed(training_seed)
    env.action_space.seed(training_seed)
    Q = np.zeros((num_states, num_actions))
    success_count = 0
    fail_count = 0
    found_success = False
    recent_successes = []
    best_success_episode = None
    epsilonsHistory = []
    alphasHistory = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=training_seed + ep + 1)
        env.action_space.seed(training_seed + ep + 1)
        if not isinstance(obs, int):
            obs = env.state_to_index(obs)
        episode = []
        total_rewards = 0
        episode_success = False

        for step in range(max_steps):
            eps = get_decayed_epsilon(ep, epsilon_decay_rate, min_epsilon, extreme, found_success)
            action = np.random.randint(Q.shape[1]) if np.random.rand() < eps else np.argmax(Q[obs])
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if not isinstance(next_obs, int):
                next_obs = env.state_to_index(next_obs)
            episode.append((obs, int(action), float(reward)))
            total_rewards += reward
            obs = next_obs
            if terminated or truncated:
                if reward > 0.0:
                    success_count += 1
                    episode_success = True
                    best_success_episode = episode[:]
                else:
                    fail_count += 1
                break

        recent_successes.append(1 if episode_success else 0)
        if len(recent_successes) > 2000:
            recent_successes.pop(0)
        if episode_success:
            found_success = True

        if extreme:
            alpha_ep = extreme_alpha(ep, found_success, episode_success, start_alpha=start_alpha,
                                           max_alpha=max_alpha, min_alpha=min_alpha, alpha_decay_rate=alpha_decay_rate)
        else:
            alpha_ep = adaptive_alpha(ep, sum(recent_successes[-success_threshold:]),
                                      start_alpha=start_alpha, max_alpha=max_alpha, min_alpha=min_alpha,
                                      success_threshold=success_threshold, alpha_decay_rate=alpha_decay_rate)

        epsilonsHistory.append(eps)
        alphasHistory.append(alpha_ep)

        G = 0
        for state, action, reward in reversed(episode):
            state = int(state)
            action = int(action)
            Q[state, action] += alpha_ep * (G - Q[state, action])
            G = reward + gamma * G

        if extreme and found_success and best_success_episode and (ep % 10000 == 0):
            G = 0
            for state, action, reward in reversed(best_success_episode):
                Q[int(state), int(action)] += alpha_ep * (G - Q[int(state), int(action)])
                G = reward + gamma * G

        all_rewards.append(total_rewards)
        if (ep % 2000 == 0 and len(all_rewards) >= 2000) or ep == (num_episodes - 1):
            avg_reward = np.mean(all_rewards[-100:])
            print(f"Episode {ep}, AvgReward(last100): {avg_reward:.3f}")
            print(f"Successes (last 2000): {success_count}   Failures: {fail_count}")
            success_count, fail_count = 0, 0

    print(f"max mean episode reward across seeds: {max(all_rewards)}")
    meta_data = {
        "title": "Learning Progress using MONTE_CARLO for DiagonalFrozenLake",
        "label": "MONTE CARLO",
        "filename": filename
    }
    plot_graph(meta_data, all_rewards, window=plot_window)

    # Plot Epsilon and Alpha curves
    plt.figure(figsize=(12, 6))
    plt.plot(epsilonsHistory, label='Epsilon', color='blue', linewidth=2)
    plt.plot(alphasHistory, label='Alpha', color='red', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Value')
    plt.title('Epsilon and Alpha schedules during Monte Carlo training')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    window = 1000
    smoothed_rewards = np.convolve(all_rewards, np.ones(window) / window, mode='valid')
    max_rewards = np.array([np.max(all_rewards[i:i+window]) for i in range(len(all_rewards) - window + 1)])
    min_rewards = np.array([np.min(all_rewards[i:i+window]) for i in range(len(all_rewards) - window + 1)])

    x_range = range(window, len(all_rewards) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(x_range[:len(smoothed_rewards)], smoothed_rewards, label="Average Reward", color="blue", linewidth=2)
    plt.plot(x_range[:len(max_rewards)], max_rewards, label="Max Reward (window)", color="green", linestyle="--", alpha=0.7)
    plt.plot(x_range[:len(min_rewards)], min_rewards, label="Min Reward (window)", color="red", linestyle="--", alpha=0.7)

    plt.fill_between(x_range[:len(max_rewards)], min_rewards, max_rewards, color='gray', alpha=0.1)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Monte Carlo Performance: Avg, Max & Min Rewards (window=1000)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    env.close()
    return Q, all_rewards, None, None

def evaluate_policy(env, Q, episodes=100, base_seed=1000):
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        if not isinstance(obs, int):
            obs = env.state_to_index(obs)
        total_reward = 0
        for _ in range(100): # Use max_steps as during training
            action = np.argmax(Q[int(obs)])
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if not isinstance(next_obs, int):
                next_obs = env.state_to_index(next_obs)
            total_reward += reward
            obs = next_obs
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards), rewards