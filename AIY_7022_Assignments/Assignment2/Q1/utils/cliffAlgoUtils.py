import numpy as np
from cliff import set_global_seed
from utils.plotUtils import plot_graph
import os
import imageio

def get_decayed_epsilon(episode, start_epsilon=1.0, decay_rate=0.997, min_epsilon=0.05):
    return max(min_epsilon, start_epsilon * (decay_rate ** episode))

def get_decayed_alpha(episode, start_alpha=0.999, decay_rate=0.9997, min_alpha=0.6):
    return max(min_alpha, start_alpha * (decay_rate ** episode)) 

def epsilon_greedy(Q, state, episode, num_episodes):
    # decaying epsilon schedule
    eps = get_decayed_epsilon(episode)
    if np.random.rand() < eps:
        return np.random.randint(Q.shape[1])  # explore
    return np.argmax(Q[state])  # exploit

def summarize_performance(rewards):
    interimMean = np.mean(rewards[:500])     # first 500
    asymptoticMean = np.mean(rewards[-1000:])  # last 1000
    asymptoticVariance = np.var(rewards[-1000:])  # last 1000
    
    print(f"Interim mean reward (first 500 episodes): {interimMean}")
    print(f"Asymptotic mean reward (last 1000 episodes): {asymptoticMean}")
    print(f"Asymptotic variance in reward (last 1000 episodes): {asymptoticVariance}")
    pass

def SARSA_UPDATE_RULE(Q, obs, action, reward, next_obs, next_action, terminated, **kwargs):
    alpha, gamma = kwargs["alpha"], kwargs["gamma"]
    Q[obs, action] += alpha * (
        reward + gamma * Q[next_obs, next_action] * (not terminated) - Q[obs, action]
    )
    return Q

def Q_UPDATE_RULE(Q, obs, action, reward, next_obs, next_action, terminated, **kwargs):
    alpha, gamma = kwargs["alpha"], kwargs["gamma"]
    best_next = np.argmax(Q[next_obs])
    Q[obs, action] += alpha * (
        reward + gamma * Q[next_obs, best_next] * (not terminated) - Q[obs, action]
    )
    return Q

def EXPECTED_SARSA_UPDATE_RULE(Q, obs, action, reward, next_obs, next_action, terminated, **kwargs):
    alpha, gamma, epsilon = kwargs["alpha"], kwargs["gamma"], kwargs["epsilon"]
    num_actions = Q.shape[1]
    # Expected SARSA update
    policy_probs = np.ones(num_actions) * (epsilon / num_actions)
    policy_probs[np.argmax(Q[next_obs])] += (1 - epsilon)
    expected_value = np.dot(Q[next_obs], policy_probs)

    Q[obs, action] += alpha * (
        reward + gamma * expected_value * (not terminated) - Q[obs, action]
    )
    return Q

def run_algorithm(env, 
                  update_rule,
                  training_seeds=[0,10,20,30,40,50,60,70,80,90], # Question 1.1 :: Train over 10 different random seeds
                  num_episodes=20000,
                  gamma=0.99, 
                  max_steps=1000,
                  filename = None,
                  window = 1
                 ):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    print(f"Number of states in the state space is :: {num_states}") 
    print(f"Number of actions in the action space is :: {num_actions}") 

    all_rewards = []
    all_safe, all_risky = [], []

    for seed in training_seeds:
        print(f"Running for seed : {seed}")
        
        # train each agent for 10 different seeds
        set_global_seed(seed)
        env.action_space.seed(seed)

        Q = np.zeros((num_states, num_actions))
        episode_rewards = []
        safe_visits, risky_visits = 0, 0

        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed+ep+1) # reset the enironment to the new seed
            env.action_space.seed(seed + ep + 1)
            action = epsilon_greedy(Q, obs, ep, num_episodes)

            total_reward = 0
            for step in range(max_steps):
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_action = epsilon_greedy(Q, next_obs, ep, num_episodes)

                # update rule (pass current epsilon for Expected SARSA)
                Q = update_rule(Q, obs, action, reward, next_obs, next_action, terminated, 
                                alpha=get_decayed_alpha(ep), gamma=gamma, epsilon=get_decayed_epsilon(ep))

                obs, action = next_obs, next_action
                total_reward += reward

                if info.get("goal") == "safe":
                    safe_visits += 1
                elif info.get("goal") == "risky":
                    risky_visits += 1

                if terminated or truncated:
                    break

            episode_rewards.append(total_reward)
            
            if ep % 5000 == 0:   # print every 5000 episode
                print(f"Total rewards after {ep} episodes : {total_reward}")

        # Question 1.2 :: for each algorithm :: store the episodic rewards across for each seed here 
        all_rewards.append(episode_rewards)
        all_safe.append(safe_visits)
        all_risky.append(risky_visits)
    
    env.close()

    # Question 1.2.1 :: average the episodic rewards accross the 10 seeds
    episode_rewards = np.mean(all_rewards, axis=0).tolist()
    summarize_performance(episode_rewards)
    print(f"max mean episode reward across seeds: {max(episode_rewards)}")
    
    # Question 1.2.2 :: Plot these average rewards v/s episodes for each algorithm :: this will show the average learning progress over time
    meta_data = get_meta_data(update_rule)
    if filename is not None:
        meta_data["filename"] = filename
    plot_graph(meta_data, episode_rewards, window = window)
    
    safe_visits = np.mean(all_safe)
    risky_visits = np.mean(all_risky)

    return Q, episode_rewards, safe_visits, risky_visits

def get_meta_data(update_rule):
    if update_rule == SARSA_UPDATE_RULE:
        return {
            "title" : "Average Learning Progress using SARSA",
            "label" : "SARSA",
            "filename" : "sarsa plot.png"
        }
    elif update_rule == Q_UPDATE_RULE:
        return {
            "title" : "Average Learning Progress using Q-Learning",
            "label" : "Q-Learning",
            "filename" : "qlearning plot.png"
        }
    else:
        return {
            "title" : "Average Learning Progress using Expected SARSA",
            "label" : "Expected SARSA",
            "filename" : "expected sarsa plot.png"
        }

def evaluate_policy(env, Q_table, episodes=100, maxSteps = 1000):
    """
    Evaluate a learned Q-table over given episodes.
    
    Args:
        env: Environment instance
        Q_table: Learned Q-table (2D: states x actions)
        episodes: Number of evaluation episodes
    
    Returns:
        avg_reward: Average reward over episodes
        safe_visits: Total safe-goal visits
        risky_visits: Total risky-goal visits
    """
    rewards = []
    safe_visits, risky_visits = 0, 0

    for i in range(episodes):
        set_global_seed(i)
        state, _ = env.reset(seed=i)   # reset with different seed
        done = False
        ep_reward = 0
        steps = 0

        while not done:
            action = np.argmax(Q_table[state])  # greedy action
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            state = next_state
            done = terminated or truncated

            # Count goal visits
            if info.get("goal") == "safe":
                safe_visits += 1
            elif info.get("goal") == "risky":
                risky_visits += 1
            
            steps += 1
            if steps > maxSteps:
                print("Couldn't reach terminal status")
                break

        rewards.append(ep_reward)

    return rewards, safe_visits, risky_visits

    
def generate_gif(env, Q, outputFile, max_steps = 1000):
    frames = []
    obs, _ = env.reset(seed=100)  # fixed seed for consistency
    
    # Render a RGB frame
    frame = env.render()
    frames.append(frame)
    
    for step in range(max_steps):
        # greedy action
        action = np.argmax(Q[obs])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render RGB frame
        frame = env.render()
        frames.append(frame)

        if terminated or truncated:
            print(info.get("goal"))
            break

    # ensure gifs folder exists
    os.makedirs("gifs", exist_ok=True)

    # save GIF
    gif_path = os.path.join("gifs", outputFile)
    imageio.mimsave(gif_path, frames, fps=4)
    print(f"Saved GIF: {gif_path}")