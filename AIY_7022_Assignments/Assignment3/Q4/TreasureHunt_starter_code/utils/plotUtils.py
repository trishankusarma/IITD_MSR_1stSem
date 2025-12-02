import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_rewards(plot_dir, model_name, rewards_per_episode):
    """
    Plot training rewards with adaptive smoothing.
    The smoothing window scales with the total number of episodes.
    """
    PLOT_PATH = os.path.join(plot_dir, f"{model_name}_training_rewards.png")

    # Adaptive smoothing window: ~1% of total length (min 20, max 500)
    smooth_window = max(20, min(len(rewards_per_episode) // 100, 500))

    # Apply smoothing
    if len(rewards_per_episode) >= smooth_window:
        rewards_smooth = np.convolve(rewards_per_episode,
                                     np.ones(smooth_window) / smooth_window,
                                     mode='valid')
    else:
        rewards_smooth = rewards_per_episode

    plt.figure(figsize=(8, 5))
    plt.plot(rewards_per_episode, alpha=0.3, label="Raw Reward", color='tab:gray')
    plt.plot(range(smooth_window - 1, len(rewards_smooth) + smooth_window - 1),
             rewards_smooth, label=f"Smoothed (window={smooth_window})", color='tab:blue', linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards (DQN + PER)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"Reward curve saved to {PLOT_PATH} (window={smooth_window})")


def plot_evaluation_curves(agent, eval_interval=1000,
                           output_path=os.path.join("plots", "evaluation_trends.png")):
    """
    Plot evaluation trends (unsmoothed) from stored mean metrics in the agent.
    X-axis shows episode checkpoints.
    """
    eval_x = [i * eval_interval for i in range(1, len(agent.eval_mean_rewards) + 1)]

    plt.figure(figsize=(9, 5))
    plt.plot(eval_x, agent.eval_mean_rewards, label='Mean Reward', linewidth=2.5, color='tab:blue')
    plt.plot(eval_x, agent.eval_mean_treasures, label='Mean Treasures', linestyle='--', linewidth=2, color='tab:green')
    plt.plot(eval_x, agent.eval_mean_pirates, label='Mean Pirates', linestyle=':', linewidth=2, color='tab:red')
    plt.plot(eval_x, agent.eval_mean_goals, label='Mean Goals', linestyle='-.', linewidth=2, color='tab:orange')

    plt.xlabel("Training Episodes")
    plt.ylabel("Evaluation Metrics")
    plt.title("Evaluation Trends across Training")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Evaluation plot saved at {output_path}")
