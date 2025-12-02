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