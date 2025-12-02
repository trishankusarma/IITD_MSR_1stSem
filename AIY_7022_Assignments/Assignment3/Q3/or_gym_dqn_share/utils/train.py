import numpy as np
import torch
from tqdm import tqdm
from utils.actions import encode_action_index
from utils.config import get_env_config
from utils.evaluate import evaluate
from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv


def train_loop(env, agent, part: str, num_episodes=10_000, max_steps=10, print_every=500, eval_every=5000, seed=42):
    """
    Trains a DQN agent on the DiscretePortfolioOptEnv.

    Args:
        env: OR-Gym environment instance.
        agent: Agent object implementing select_action, optimize_model, update_target_network, buffer.
        part: 'a' or 'b' for different reward shaping schemes.
        num_episodes: Number of training episodes.
        max_steps: Maximum steps per episode.
        print_every: Print average return every N episodes.
        eval_every: Run evaluation every N episodes.
        seed: Base seed for reproducibility.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    episode_returns, losses = [], []
    opt_steps = 0

    for ep in tqdm(range(num_episodes), desc="Training Episodes"):
        # Handle environments returning (obs, info)
        reset_out = env.reset()
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        episode_reward = 0.0
        eps = agent.get_epsilon()

        for step in range(max_steps):
            # Select action
            a_idx = agent.select_action(state, eps)
            a_vec = encode_action_index(a_idx)

            # Compute previous wealth
            cash, prices, holdings = state[0], state[1:1 + env.num_assets], state[1 + env.num_assets:1 + 2 * env.num_assets]
            prev_wealth = cash + np.dot(prices, holdings)

            # Step environment
            next_state, reward, done, _ = env.step(a_vec)
            if isinstance(next_state, tuple):  # safety for OR-Gym updates
                next_state = next_state[0]
            cash_next = next_state[0]
            prices_next = next_state[1:1 + env.num_assets]
            holdings_next = next_state[1 + env.num_assets:1 + 2 * env.num_assets]
            curr_wealth = cash_next + np.dot(prices_next, holdings_next)

            # Reward shaping
            if part.lower() == "a":
                reward_shaped = reward if done else 0.0
            else:
                ratio = (curr_wealth + 1e-8) / (prev_wealth + 1e-8)
                safe_ratio = np.maximum(ratio, 1e-8)  # avoid log(0) or log(negative)
                log_ratio = np.log(safe_ratio)
                reward_shaped = log_ratio - 0.5 * (log_ratio ** 2)

            # Store transition
            agent.buffer.push(state, a_idx, reward_shaped, next_state, done)
            state = next_state
            episode_reward += reward  # environment reward, not shaped

            # Optimize model if possible
            loss = agent.optimize_model()
            if loss is not None:
                if torch.is_tensor(loss):
                    loss = loss.item()
                losses.append(loss)
                opt_steps += 1

            # Periodic target update
            if opt_steps >= agent.target_update_freq and opt_steps % agent.target_update_freq == 0:
                agent.update_target_network()

            if done:
                break

        episode_returns.append(episode_reward)

        # Logging
        if (ep + 1) % print_every == 0:
            avg_ret = np.mean(episode_returns[-print_every:])
            print(f"[Ep {ep+1:5d}] avg_return={avg_ret:.3f}  ε={eps:.3f}  buf={len(agent.buffer)}  opt={opt_steps}")

        # Periodic evaluation
        if (ep + 1) % eval_every == 0:
            title = f"Task {part.upper()} Terminal Wealth"
            save_dir = f"evaluation_part_{part.upper()}_ep{ep+1}"
            evaluate(
                agent,
                lambda seed_: DiscretePortfolioOptEnv(env_config=get_env_config(seed_)),
                num_seeds=50,
                max_steps=max_steps,
                title=title,
                save_dir=save_dir
            )

    return episode_returns, losses