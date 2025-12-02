import matplotlib.pyplot as plt
import numpy as np

def save_gif(frames, filename, duration=50):
    """Save frames as GIF"""
    try:
        from PIL import Image
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(filename, save_all=True, append_images=images[1:], 
                      duration=duration, loop=0)
        print(f"Saved: {filename}")
    except ImportError:
        print("PIL not available, skipping GIF generation")

def generatePlots(actor_losses, critic_losses, episode_rewards, filename = "plots/a2c_training_curves.png"):
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Actor Loss
    axes[0].plot(actor_losses, label="Actor Loss", alpha=0.6, linewidth=0.5)
    if len(actor_losses) > 100:
        smoothed = np.convolve(actor_losses, np.ones(100)/100, mode='valid')
        axes[0].plot(range(99, len(actor_losses)), smoothed, label="Smoothed (100)", linewidth=2)
    axes[0].set_title("Actor Loss")
    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Critic Loss
    axes[1].plot(critic_losses, label="Critic Loss", alpha=0.6, linewidth=0.5)
    if len(critic_losses) > 100:
        smoothed = np.convolve(critic_losses, np.ones(100)/100, mode='valid')
        axes[1].plot(range(99, len(critic_losses)), smoothed, label="Smoothed (100)", linewidth=2)
    axes[1].set_title("Critic Loss")
    axes[1].set_xlabel("Training Steps")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Episode Rewards
    axes[2].plot(episode_rewards, label="Episode Reward", color='orange', alpha=0.6, linewidth=0.5)
    if len(episode_rewards) > 10:
        smoothed = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
        axes[2].plot(range(9, len(episode_rewards)), smoothed, label="Moving Average (10)", 
                     color='red', linewidth=2)
    axes[2].axhline(y=200, color='green', linestyle='--', label='Target (200)', linewidth=2)
    axes[2].set_title("Episode Rewards")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Reward")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{filename}", dpi=150)
    print(f"Saved: {filename}")
    plt.close()