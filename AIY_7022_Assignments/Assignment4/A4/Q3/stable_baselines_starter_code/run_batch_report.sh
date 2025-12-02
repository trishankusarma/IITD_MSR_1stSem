"""
Analysis script to extract TensorBoard data and generate comparison plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
from scipy import stats
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def extract_tensorboard_data(logdir, tag='rollout/ep_rew_mean'):
    """
    Extract scalar data from TensorBoard logs
    
    Args:
        logdir: Path to TensorBoard log directory
        tag: Tag to extract (default: episode reward mean)
    
    Returns:
        DataFrame with columns: step, value
    """
    try:
        ea = event_accumulator.EventAccumulator(logdir)
        ea.Reload()
        
        if tag not in ea.Tags()['scalars']:
            print(f"Warning: Tag '{tag}' not found in {logdir}")
            return None
        
        data = ea.Scalars(tag)
        df = pd.DataFrame(data)
        df = df[['step', 'value']]
        return df
    except Exception as e:
        print(f"Error reading {logdir}: {e}")
        return None


def find_tensorboard_logs(base_dir, algo, env, seed):
    """Find the actual tensorboard log directory"""
    log_path = os.path.join(base_dir, algo, env, f"seed_{seed}")
    
    if not os.path.exists(log_path):
        return None
    
    # Find subdirectory (e.g., a2c_42_1)
    subdirs = [d for d in os.listdir(log_path) if os.path.isdir(os.path.join(log_path, d))]
    if subdirs:
        return os.path.join(log_path, subdirs[0])
    
    return log_path


def load_all_runs(base_dir='logs', envs=None, algos=None, seeds=None):
    """
    Load all training runs
    
    Returns:
        Dictionary: {env: {algo: {seed: dataframe}}}
    """
    if envs is None:
        envs = ['InvertedPendulum-v4', 'Hopper-v4', 'HalfCheetah-v4']
    if algos is None:
        algos = ['a2c', 'ppo']
    if seeds is None:
        seeds = [42, 123, 456]
    
    data = {}
    
    for env in envs:
        data[env] = {}
        for algo in algos:
            data[env][algo] = {}
            for seed in seeds:
                log_dir = find_tensorboard_logs(base_dir, algo, env, seed)
                if log_dir:
                    df = extract_tensorboard_data(log_dir)
                    if df is not None:
                        data[env][algo][seed] = df
                        print(f"✓ Loaded: {algo}/{env}/seed_{seed}")
                    else:
                        print(f"✗ Failed: {algo}/{env}/seed_{seed}")
    
    return data


def plot_learning_curves(data, env, save_path=None):
    """
    Plot learning curves for both algorithms on a single environment
    
    Args:
        data: Dictionary from load_all_runs
        env: Environment name
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'a2c': '#2E86AB', 'ppo': '#A23B72'}
    
    for algo in ['a2c', 'ppo']:
        if algo not in data[env] or not data[env][algo]:
            continue
        
        # Collect data from all seeds
        all_dfs = list(data[env][algo].values())
        
        if not all_dfs:
            continue
        
        # Find minimum length across seeds
        min_len = min(len(df) for df in all_dfs)
        
        # Truncate all to same length and stack
        rewards = np.array([df['value'].values[:min_len] for df in all_dfs])
        steps = all_dfs[0]['step'].values[:min_len]
        
        # Compute mean and std
        mean_rewards = rewards.mean(axis=0)
        std_rewards = rewards.std(axis=0)
        
        # Plot
        ax.plot(steps, mean_rewards, label=algo.upper(), color=colors[algo], linewidth=2)
        ax.fill_between(steps, 
                        mean_rewards - std_rewards, 
                        mean_rewards + std_rewards, 
                        color=colors[algo],
                        alpha=0.2)
    
    ax.set_xlabel('Timesteps', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title(f'Learning Curves: {env}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    
    plt.show()


def plot_all_environments(data, save_dir='plots'):
    """Generate learning curve plots for all environments"""
    os.makedirs(save_dir, exist_ok=True)
    
    for env in data.keys():
        save_path = os.path.join(save_dir, f'{env}_learning_curves.png')
        plot_learning_curves(data, env, save_path)


def compute_final_performance(data, window=10):
    """
    Compute final performance statistics
    
    Args:
        data: Dictionary from load_all_runs
        window: Number of final evaluations to average
    
    Returns:
        DataFrame with results
    """
    results = []
    
    for env in data.keys():
        for algo in data[env].keys():
            final_rewards = []
            
            for seed, df in data[env][algo].items():
                if len(df) >= window:
                    # Average over last 'window' evaluations
                    final_reward = df['value'].values[-window:].mean()
                    final_rewards.append(final_reward)
            
            if final_rewards:
                results.append({
                    'Environment': env,
                    'Algorithm': algo.upper(),
                    'Mean Reward': np.mean(final_rewards),
                    'Std Reward': np.std(final_rewards),
                    'Min Reward': np.min(final_rewards),
                    'Max Reward': np.max(final_rewards),
                    'N Seeds': len(final_rewards)
                })
    
    df_results = pd.DataFrame(results)
    return df_results


def statistical_comparison(data, window=10):
    """
    Perform statistical comparison between A2C and PPO
    
    Args:
        data: Dictionary from load_all_runs
        window: Number of final evaluations to average
    
    Returns:
        DataFrame with comparison results
    """
    comparisons = []
    
    for env in data.keys():
        # Get final rewards for both algorithms
        a2c_rewards = []
        ppo_rewards = []
        
        if 'a2c' in data[env]:
            for seed, df in data[env]['a2c'].items():
                if len(df) >= window:
                    a2c_rewards.append(df['value'].values[-window:].mean())
        
        if 'ppo' in data[env]:
            for seed, df in data[env]['ppo'].items():
                if len(df) >= window:
                    ppo_rewards.append(df['value'].values[-window:].mean())
        
        if len(a2c_rewards) > 0 and len(ppo_rewards) > 0:
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(a2c_rewards, ppo_rewards)
            
            # Determine winner
            a2c_mean = np.mean(a2c_rewards)
            ppo_mean = np.mean(ppo_rewards)
            
            if p_value < 0.05:
                winner = 'PPO' if ppo_mean > a2c_mean else 'A2C'
                significant = 'Yes'
            else:
                winner = 'No significant difference'
                significant = 'No'
            
            comparisons.append({
                'Environment': env,
                'A2C Mean': f"{a2c_mean:.2f}",
                'PPO Mean': f"{ppo_mean:.2f}",
                'Difference': f"{ppo_mean - a2c_mean:.2f}",
                't-statistic': f"{t_stat:.3f}",
                'p-value': f"{p_value:.4f}",
                'Significant (p<0.05)': significant,
                'Winner': winner
            })
    
    df_comparison = pd.DataFrame(comparisons)
    return df_comparison


def create_summary_report(data, output_path='results_summary.txt'):
    """
    Create a text summary report
    
    Args:
        data: Dictionary from load_all_runs
        output_path: Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STABLE BASELINES3: A2C vs PPO PERFORMANCE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Final performance
        f.write("FINAL PERFORMANCE (Last 10 Evaluations Average)\n")
        f.write("-" * 80 + "\n")
        df_perf = compute_final_performance(data)
        f.write(df_perf.to_string(index=False))
        f.write("\n\n")
        
        # Statistical comparison
        f.write("STATISTICAL COMPARISON\n")
        f.write("-" * 80 + "\n")
        df_comp = statistical_comparison(data)
        f.write(df_comp.to_string(index=False))
        f.write("\n\n")
        
        # Per-environment analysis
        for env in data.keys():
            f.write(f"\n{env}\n")
            f.write("=" * 80 + "\n")
            
            for algo in ['a2c', 'ppo']:
                if algo in data[env] and data[env][algo]:
                    f.write(f"\n{algo.upper()}:\n")
                    
                    for seed, df in data[env][algo].items():
                        if len(df) > 0:
                            final_reward = df['value'].values[-10:].mean()
                            max_reward = df['value'].max()
                            f.write(f"  Seed {seed}: Final = {final_reward:.2f}, Max = {max_reward:.2f}\n")
            
            f.write("\n")
    
    print(f"\nSummary report saved to: {output_path}")


def main():
    """Main analysis pipeline"""
    print("=" * 80)
    print("LOADING TENSORBOARD DATA")
    print("=" * 80)
    
    # Load all data
    data = load_all_runs()
    
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    # Generate plots
    plot_all_environments(data)
    
    print("\n" + "=" * 80)
    print("COMPUTING STATISTICS")
    print("=" * 80)
    
    # Print final performance
    print("\nFINAL PERFORMANCE:")
    df_perf = compute_final_performance(data)
    print(df_perf.to_string(index=False))
    
    # Print statistical comparison
    print("\n\nSTATISTICAL COMPARISON:")
    df_comp = statistical_comparison(data)
    print(df_comp.to_string(index=False))
    
    # Create summary report
    create_summary_report(data)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nOutputs:")
    print("  - Learning curve plots: plots/")
    print("  - Summary report: results_summary.txt")


if __name__ == "__main__":
    main()