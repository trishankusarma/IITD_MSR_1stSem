import os
import numpy as np
import matplotlib.pyplot as plt

def plot_training_curve(episode_rewards, baseline_type):
    """Plot training curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, alpha=0.3, label='Episode Return')
    
    if len(episode_rewards) >= 100:
        moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        plt.plot(range(99, len(episode_rewards)), moving_avg, 
                label='100-episode Moving Average', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(f'Training Progress: {baseline_type}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/all_rewards_{baseline_type}.png", dpi=150)
    plt.close()
