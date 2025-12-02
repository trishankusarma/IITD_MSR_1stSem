import numpy as np
import matplotlib.pyplot as plt
from utils.actions import encode_action_index
from utils.config import set_seed
import os
import json

def evaluate(agent, env_fn, num_seeds=100, max_steps=10, title="Evaluation",
             save_dir="evaluation", training_rewards=None, training_losses=None):
    """
    Evaluate agent on multiple seeds and generate required plots.
   
    Args:
        agent: Trained DQN agent
        env_fn: Function that creates environment given a seed
        num_seeds: Number of seeds to evaluate (default 100)
        max_steps: Maximum steps per episode (default 10)
        title: Title prefix for plots
        save_dir: Directory to save plots
        training_rewards: List of rewards per episode during training (optional)
        training_losses: List of losses per optimization step during training (optional)
   
    Returns:
        Dictionary with evaluation metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    all_wealths = []

    for seed in range(num_seeds):
        env = env_fn(seed)
        set_seed(seed)
        obs = env.reset()

        # Initial wealth = cash + Σ(price × holdings)
        wealth = [obs[0] + np.sum(obs[1:1+env.num_assets] * obs[1+env.num_assets:1+2*env.num_assets])]

        for _ in range(max_steps):
            a_idx = agent.select_action(obs, eps=0.0)  # pure greedy
            obs, _, done, _ = env.step(encode_action_index(a_idx))
            wealth.append(obs[0] + np.sum(obs[1:1+env.num_assets] * obs[1+env.num_assets:1+2*env.num_assets]))
            if done:
                break

        # Pad if shorter
        if len(wealth) < (max_steps + 1):
            wealth += [wealth[-1]] * (max_steps + 1 - len(wealth))

        all_wealths.append(wealth)
        env.close()

    # Convert to array: shape (num_seeds, max_steps+1)
    arr = np.array(all_wealths)
    mean_wealth = np.mean(arr, axis=0)
    std_wealth = np.std(arr, axis=0)
    var_wealth = np.var(arr, axis=0)
    steps = np.arange(max_steps + 1)

    # ============ PLOT 1: Mean Portfolio Wealth with Std Deviation ============
    plt.figure(figsize=(10, 6))
    plt.plot(steps, mean_wealth, 'b-', linewidth=2, label='Mean Wealth')
    plt.fill_between(steps, mean_wealth - std_wealth, mean_wealth + std_wealth,
                     alpha=0.3, color='blue', label='± 1 Std Dev')
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Portfolio Wealth", fontsize=12)
    plt.title(f"{title}: Mean Portfolio Wealth (± Std) over {num_seeds} Seeds", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{title.lower().replace(' ', '_')}_wealth_mean_std.png", dpi=150)
    plt.close()
    print(f"✓ Saved: {save_dir}/{title.lower().replace(' ', '_')}_wealth_mean_std.png")

    # ============ PLOT 2: Individual Trajectories (Optional - shows variability) ============
    plt.figure(figsize=(10, 6))
    # Plot a subset of trajectories for clarity
    num_show = min(20, num_seeds)
    for i in range(num_show):
        plt.plot(steps, arr[i], alpha=0.3, linewidth=0.8)
    plt.plot(steps, mean_wealth, 'r-', linewidth=2.5, label='Mean Wealth')
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Portfolio Wealth", fontsize=12)
    plt.title(f"{title}: Sample Wealth Trajectories", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{title.lower().replace(' ', '_')}_trajectories.png", dpi=150)
    plt.close()
    print(f"✓ Saved: {save_dir}/{title.lower().replace(' ', '_')}_trajectories.png")

    # ============ PLOT 3: Standard Deviation over Time ============
    plt.figure(figsize=(10, 6))
    plt.plot(steps, std_wealth, 'g-', linewidth=2, marker='o')
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Standard Deviation of Wealth", fontsize=12)
    plt.title(f"{title}: Wealth Standard Deviation over Time", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{title.lower().replace(' ', '_')}_std_over_time.png", dpi=150)
    plt.close()
    print(f"✓ Saved: {save_dir}/{title.lower().replace(' ', '_')}_std_over_time.png")

    # ============ PLOT 4: Training Rewards (if provided) ============
    if training_rewards is not None and len(training_rewards) > 0:
        plt.figure(figsize=(10, 6))
       
        episodes = np.arange(len(training_rewards))
       
        # Smoothed rewards with moving average
        window = 100  # 100-episode moving average
        if len(training_rewards) >= window:
            smoothed = np.convolve(training_rewards, np.ones(window)/window, mode='valid')
            smoothed_episodes = episodes[window-1:]
           
            plt.plot(episodes, training_rewards, alpha=0.2, linewidth=0.5, color='blue', label='Raw Rewards')
            plt.plot(smoothed_episodes, smoothed, linewidth=2, color='darkblue', label=f'{window}-Episode Moving Average')
        else:
            plt.plot(episodes, training_rewards, linewidth=2, color='blue', label='Rewards')
       
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Total Reward", fontsize=12)
        plt.title(f"{title}: Training Rewards over Episodes", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{title.lower().replace(' ', '_')}_training_rewards.png", dpi=150)
        plt.close()
        print(f"✓ Saved: {save_dir}/{title.lower().replace(' ', '_')}_training_rewards.png")

    # ============ PLOT 5: Training Loss per Episode (if provided) ============
    if training_losses is not None and len(training_losses) > 0:
        plt.figure(figsize=(10, 6))
       
        num_episodes = len(training_rewards) if training_rewards is not None else len(training_losses)
    
        if len(training_losses) > num_episodes:
            # Multiple optimization steps per episode - aggregate
            losses_per_step = len(training_losses) // num_episodes
            episode_losses = []
            for i in range(num_episodes):
                start_idx = i * losses_per_step
                end_idx = min((i + 1) * losses_per_step, len(training_losses))
                episode_losses.append(np.mean(training_losses[start_idx:end_idx]))
            episode_losses = np.array(episode_losses)
        else:
            # Assume one loss per episode or pad
            episode_losses = np.array(training_losses[:num_episodes])
       
        episodes = np.arange(len(episode_losses))
       
        # Smoothed loss with moving average
        window = 100  # 100-episode moving average
        if len(episode_losses) >= window:
            smoothed = np.convolve(episode_losses, np.ones(window)/window, mode='valid')
            smoothed_episodes = episodes[window-1:]
           
            plt.plot(episodes, episode_losses, alpha=0.2, linewidth=0.5, color='red', label='Raw Loss')
            plt.plot(smoothed_episodes, smoothed, linewidth=2, color='darkred', label=f'{window}-Episode Moving Average')
        else:
            plt.plot(episodes, episode_losses, linewidth=2, color='red', label='Loss')
       
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title(f"{title}: Training Loss over Episodes", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{title.lower().replace(' ', '_')}_training_loss.png", dpi=150)
        plt.close()
        print(f"✓ Saved: {save_dir}/{title.lower().replace(' ', '_')}_training_loss.png")

    # ============ Calculate Metrics ============
    # Final timestep (t=10) ratio
    final_mean = mean_wealth[-1]
    final_std = std_wealth[-1]
    final_var = var_wealth[-1]
    final_ratio = final_mean / (final_std + 1e-8)
   
    # Overall ratio (mean across all timesteps and seeds)
    overall_mean = np.mean(arr)
    overall_std = np.std(arr)
    overall_var = np.var(arr)
    overall_ratio = overall_mean / (overall_std + 1e-8)

    # ============ Print Summary ============
    print("\n" + "="*70)
    print(f"{title} - EVALUATION RESULTS ({num_seeds} seeds)")
    print("="*70)
    print(f"Mean Total Wealth at t=10       : {final_mean:.4f}")
    print(f"Std of Total Wealth at t=10     : {final_std:.4f}")
    print(f"Variance of Total Wealth at t=10: {final_var:.4f}")
    print(f"Ratio (Mean/Std) at t=10        : {final_ratio:.4f}")
    print(f"-"*70)
    print(f"Overall Mean (all steps & seeds): {overall_mean:.4f}")
    print(f"Overall Std (all steps & seeds) : {overall_std:.4f}")
    print(f"Overall Var (all steps & seeds) : {overall_var:.4f}")
    print(f"Overall Ratio (Mean/Std)        : {overall_ratio:.4f}")
    print("="*70 + "\n")

    # ============ Save Wealth Stats in Compact Format (Like Example) ============
    wealth_stats = {
        "all_wealths": arr.tolist(),  # All trajectories
        "mean_wealth": mean_wealth.tolist(),  # Mean at each timestep
        "std_wealth": std_wealth.tolist(),  # Std at each timestep
        "mean_terminal": float(final_mean),  # Mean at final timestep
        "std_terminal": float(final_std),  # Std at final timestep
        "ratio": float(final_ratio)  # Ratio at final timestep
    }
    
    wealth_stats_filename = f"{save_dir}/{title.lower().replace(' ', '_')}_wealth_stats.json"
    with open(wealth_stats_filename, 'w') as f:
        json.dump(wealth_stats, f, indent=4)
    print(f"✓ Saved wealth stats: {wealth_stats_filename}")