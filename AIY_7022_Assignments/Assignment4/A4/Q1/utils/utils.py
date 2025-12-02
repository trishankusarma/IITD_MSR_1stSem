import imageio
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# GIF Generation
def generate_gif(env, agent, filename, max_steps=1000, seed=42):
    """Generate a GIF of the agent's behavior in the environment."""
    frames = []
    state, _ = env.reset(seed=seed)
    done, step = False, 0
    
    while not done and step < max_steps:
        frame = env.render()
        frames.append(frame)
        action = agent.select_action(state, greedy=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        done = terminated or truncated
        step += 1
    
    imageio.mimsave(filename, frames, fps=30)
    print(f"GIF saved to {filename}")

# Smoothing Utility
def smooth(y, window_size=100):
    """Smooth data using a simple moving average."""
    y = np.array(y)
    if len(y) < window_size:
        return y
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')

# Training Curves Plot
def plot_training_curves(
    dqn_rewards=None,
    ddqn_rewards=None,
    dqn_losses=None,
    ddqn_losses=None,
    window_size=50,
    save_dir="plots",
    show=False
):
    """
    Plot DQN and Double DQN training curves.
    
    Creates:
    1. reward_curves.png - Combined reward plot (REQUIRED by assignment)
    2. Individual loss plots (optional)
    
    Args:
        dqn_rewards: Episode rewards for DQN
        ddqn_rewards: Episode rewards for Double DQN
        dqn_losses: Training losses for DQN (optional)
        ddqn_losses: Training losses for Double DQN (optional)
        window_size: Window for moving average smoothing
        save_dir: Directory to save plots
        show: Whether to display plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Combined Reward Curve
    if dqn_rewards is not None and ddqn_rewards is not None:
        plt.figure(figsize=(12, 6))
        
        episodes_dqn = np.arange(len(dqn_rewards))
        episodes_ddqn = np.arange(len(ddqn_rewards))
        
        # Plot raw data (transparent)
        plt.plot(episodes_dqn, dqn_rewards, color='skyblue', alpha=0.2, linewidth=0.5)
        plt.plot(episodes_ddqn, ddqn_rewards, color='lightcoral', alpha=0.2, linewidth=0.5)
        
        # Plot smoothed data (prominent)
        dqn_smooth = smooth(dqn_rewards, window_size)
        ddqn_smooth = smooth(ddqn_rewards, window_size)
        
        episodes_dqn_smooth = episodes_dqn[:len(dqn_smooth)]
        episodes_ddqn_smooth = episodes_ddqn[:len(ddqn_smooth)]
        
        plt.plot(episodes_dqn_smooth, dqn_smooth, color='blue', linewidth=2.5, 
                 label=f'DQN (smoothed, window={window_size})')
        plt.plot(episodes_ddqn_smooth, ddqn_smooth, color='red', linewidth=2.5, 
                 label=f'Double DQN (smoothed, window={window_size})')
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Episode Return', fontsize=12)
        plt.title('Training Rewards: DQN vs Double DQN', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        reward_plot_path = os.path.join(save_dir, "reward_curves.png")
        plt.savefig(reward_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined reward curves to {reward_plot_path}")
        
        if show:
            plt.show()
        plt.close()
    
    # Individual Loss Plots
    if dqn_losses is not None:
        plt.figure(figsize=(10, 6))
        steps = np.arange(len(dqn_losses))
        plt.plot(steps, dqn_losses, color='lightgreen', alpha=0.3, linewidth=0.5)
        
        dqn_loss_smooth = smooth(dqn_losses, window_size)
        steps_smooth = steps[:len(dqn_loss_smooth)]
        plt.plot(steps_smooth, dqn_loss_smooth, color='green', linewidth=2,
                 label=f'DQN Loss (smoothed, w={window_size})')
        
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('DQN Training Loss', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        loss_path = os.path.join(save_dir, "dqn_loss.png")
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved DQN loss curve to {loss_path}")
        
        if show:
            plt.show()
        plt.close()
    
    if ddqn_losses is not None:
        plt.figure(figsize=(10, 6))
        steps = np.arange(len(ddqn_losses))
        plt.plot(steps, ddqn_losses, color='orange', alpha=0.3, linewidth=0.5)
        
        ddqn_loss_smooth = smooth(ddqn_losses, window_size)
        steps_smooth = steps[:len(ddqn_loss_smooth)]
        plt.plot(steps_smooth, ddqn_loss_smooth, color='darkorange', linewidth=2,
                 label=f'Double DQN Loss (smoothed, w={window_size})')
        
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Double DQN Training Loss', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        loss_path = os.path.join(save_dir, "double_dqn_loss.png")
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        print(f"Saved Double DQN loss curve to {loss_path}")
        
        if show:
            plt.show()
        plt.close()

# Evaluation Results Storage
def store_evaluation_results(dqn_eval=None, ddqn_eval=None, save_dir="evaluation", 
                            save_file_name="evaluation_results.json", minimal=True):
    """
    Store evaluation results to JSON file.
    
    Args:
        dqn_eval: Dictionary with DQN evaluation stats
        ddqn_eval: Dictionary with Double DQN evaluation stats
        save_dir: Directory to save results
        save_file_name: Output filename
        minimal: If True, only save mean and std (as per assignment requirement)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if minimal:
        # Assignment asks for mean and std only
        evaluation_results = {
            "DQN": {
                "mean_return": dqn_eval.get("mean_return"),
                "std_return": dqn_eval.get("std_return")
            },
            "DOUBLE_DQN": {
                "mean_return": ddqn_eval.get("mean_return"),
                "std_return": ddqn_eval.get("std_return")
            }
        }
    else:
        # Full results (includes max, min, all episodes)
        evaluation_results = {
            "DQN": dqn_eval,
            "DOUBLE_DQN": ddqn_eval
        }
    
    output_path = os.path.join(save_dir, save_file_name)
    with open(output_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)
    
    print(f"Evaluation results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"DQN        : Mean = {dqn_eval['mean_return']:.2f} ± {dqn_eval['std_return']:.2f}")
    print(f"Double DQN : Mean = {ddqn_eval['mean_return']:.2f} ± {ddqn_eval['std_return']:.2f}")
    print("="*60)

# Q-values Per Action Plot
def plot_q_values_per_action(dqn_qvalues=None, ddqn_qvalues=None, save_dir="plots", smooth_window=20):
    """
    Plot Q-values for each action in a 2x2 grid (4 actions in LunarLander).
    
    Args:
        dqn_qvalues: numpy array of shape (timesteps, 4) with DQN Q-values
        ddqn_qvalues: numpy array of shape (timesteps, 4) with Double DQN Q-values
        save_dir: Directory to save plot
        smooth_window: Window size for smoothing Q-values
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if dqn_qvalues is None or ddqn_qvalues is None:
        print("Error: Both dqn_qvalues and ddqn_qvalues must be provided")
        return
    
    dqn_qvalues = np.array(dqn_qvalues)
    ddqn_qvalues = np.array(ddqn_qvalues)
    
    actions = ["Do Nothing (0)", "Left Engine (1)", "Main Engine (2)", "Right Engine (3)"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i in range(4):
        # Smooth Q-values
        dqn_smooth = smooth(dqn_qvalues[:, i], smooth_window)
        ddqn_smooth = smooth(ddqn_qvalues[:, i], smooth_window)
        
        timesteps_dqn = np.arange(len(dqn_smooth))
        timesteps_ddqn = np.arange(len(ddqn_smooth))
        
        # Plot DQN (blue)
        axes[i].plot(timesteps_dqn, dqn_smooth, color='blue', linewidth=2.5, 
                    alpha=0.9, label="DQN", zorder=3)
        
        # Plot Double DQN (orange/red dashed)
        axes[i].plot(timesteps_ddqn, ddqn_smooth, color='#FF7F0E', linestyle='--', 
                    linewidth=2.5, alpha=0.9, label="Double DQN", zorder=4)
        
        # Formatting
        axes[i].set_title(f"Action {i}: {actions[i]}", fontsize=12, fontweight='bold')
        axes[i].set_xlabel("Timestep", fontsize=10)
        axes[i].set_ylabel("Q-value", fontsize=10)
        axes[i].grid(True, linestyle='--', alpha=0.4)
        axes[i].legend(loc='best', fontsize=9, framealpha=0.9)
    
    plt.suptitle("Q-value Comparison per Action: DQN vs Double DQN", 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    save_path = os.path.join(save_dir, "q_values_per_action.png")
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"Saved Q-value comparison to {save_path}")
    
    if show := False:  # Set to True if you want to display
        plt.show()
    plt.close()
