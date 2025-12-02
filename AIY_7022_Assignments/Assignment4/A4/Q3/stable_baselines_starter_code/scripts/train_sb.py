"""
Stable Baselines3 Training Script for A2C and PPO
Supports InvertedPendulum-v4, Hopper-v4, and HalfCheetah-v4
"""
import argparse
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ADDITIONAL PARAMETERS
N_EVAL_EPISODES = 100

def get_hyperparameters(env_name, algo):
    """
    Returns optimized hyperparameters for each environment-algorithm pair
    """
    hyperparams = {
        'InvertedPendulum-v4': {
            'a2c': {
                'learning_rate': 7e-4,
                'n_steps': 5,
                'gamma': 0.99,
                'gae_lambda': 1.0,
                'ent_coef': 0.0,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'normalize_advantage': False,
                'total_timesteps': 300000,
            },
            'ppo': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.0,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'total_timesteps': 300000,
            }
        },
        'Hopper-v4': {
            'a2c': {
                'learning_rate': 7e-4,
                'n_steps': 8,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'ent_coef': 0.0,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'normalize_advantage': True,
                'total_timesteps': 300000,
            },
            'ppo': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.0,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'total_timesteps': 300000,
            }
        },
        'HalfCheetah-v4': {
            'a2c': {
                'learning_rate': 7e-4,
                'n_steps': 8,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'ent_coef': 0.0,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'normalize_advantage': True,
                'total_timesteps': 300000,
            },
            'ppo': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95, # smoothening factor for advantage function
                'clip_range': 0.2,
                'ent_coef': 0.0, # entropy coefficient
                'vf_coef': 0.5, # L = L_policy + v_coef*L_value - ent_coef*Entropy
                'max_grad_norm': 0.5,
                'total_timesteps': 300000,
            }
        }
    }
    
    return hyperparams[env_name][algo]

from stable_baselines3.common.callbacks import BaseCallback

class TrainLoggerCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.last_step = 0

    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step - self.last_step >= 10000:
            with open(self.log_path, "a") as f:
                f.write(f"Progress: {step} timesteps completed.\n")
            self.last_step = step
        return True

def make_env(env_name, seed):
    """
    Create and wrap the environment
    """
    def _init():
        env = gym.make(env_name)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init

def train_agent(env_name, algo, seed, device, save_dir):
    """
    Train an RL agent using specified algorithm and environment
    
    Args:
        env_name: Name of the Gym environment
        algo: Algorithm to use ('a2c' or 'ppo')
        seed: Random seed for reproducibility
        device: Device to use for training ('cuda', 'cpu', or 'auto')
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get hyperparameters
    hyperparams = get_hyperparameters(env_name, algo)
    total_timesteps = hyperparams.pop('total_timesteps')
    
    # Create directories
    log_dir = f"logs/{algo}/{env_name}/seed_{seed}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    train_log_path = os.path.join(save_dir, "training_log.txt")
    with open(train_log_path, "w") as f:
        f.write(f"Training Log\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Algorithm: {algo}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Total Timesteps: {total_timesteps}\n")
        f.write("Hyperparameters:\n")
        for k, v in hyperparams.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nStarting training...\n")
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(env_name, seed)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create eval environment
    eval_env = DummyVecEnv([make_env(env_name, seed + 1000)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix=f"{algo}_model"
    )
    
    # Create model
    if algo == 'a2c':
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            device=device,
            **hyperparams
        )
    elif algo == 'ppo':
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            device=device,
            **hyperparams
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    # Train the model
    print(f"\nTraining {algo.upper()} on {env_name} with seed {seed}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Hyperparameters: {hyperparams}\n")

    train_logger = TrainLoggerCallback(train_log_path)
    combined_cb = CallbackList([eval_callback, checkpoint_callback, train_logger])

    model.learn(
        total_timesteps=total_timesteps,
        callback=combined_cb,
        tb_log_name=f"{algo}_{seed}"
    )


    # Save final model and normalization stats
    model.save(f"{save_dir}/final_model")
    env.save(f"{save_dir}/vec_normalize.pkl")
    
    print(f"\nTraining completed! Model saved to {save_dir}")
    
    return model

def evaluate_agent(env_name, algo, seed, model_dir, n_episodes=10, render=False):

    model_path = os.path.join(model_dir, "final_model.zip")
    vec_normalize_path = os.path.join(model_dir, "vec_normalize.pkl")

    print(f"Loading model: {model_path}")
    print(f"Loading VecNormalize: {vec_normalize_path}")

    # Load model
    if algo == "a2c":
        model = A2C.load(model_path)
    else:
        model = PPO.load(model_path)

    # Load VecNormalize
    if os.path.exists(vec_normalize_path):
        env = DummyVecEnv([lambda: gym.make(env_name)])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    else:
        env = gym.make(env_name)
    
    # After loading env
    is_vec_env = isinstance(env, VecNormalize)

    episode_rewards = []
    for episode in range(n_episodes):
        if is_vec_env:
            obs = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                done = bool(dones[0])
                episode_reward += float(rewards[0])

        else:
            obs, _ = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0.0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # SAVE LOGS
    txt_path = os.path.join(model_dir, "evaluation_results.txt")

    # Save text summary
    with open(txt_path, "w") as f:
        f.write(f"Algorithm: {algo}\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Seed: {seed}\n\n")
        f.write("Episode Rewards:\n")
        for i, r in enumerate(episode_rewards):
            f.write(f"  Episode {i+1}: {r}\n")
        f.write(f"\nMean Reward: {mean_reward:.2f}\n")
        f.write(f"Std Reward: {std_reward:.2f}\n")

    print(f"\nSaved evaluation summary → {txt_path}")
    return episode_rewards

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agents with Stable Baselines3')
    parser.add_argument('--env_name', type=str, required=True,
                        choices=['InvertedPendulum-v4', 'Hopper-v4', 'HalfCheetah-v4'],
                        help='Environment name')
    parser.add_argument('--algo', type=str, default='ppo',
                        choices=['a2c', 'ppo'],
                        help='Algorithm to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate'],
                        help='Mode: train or evaluate')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Path to model for evaluation')
    parser.add_argument("--save_dir", type=str, default=None, 
                        help="Where to save model.zip")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.mode == 'train':
        train_agent(args.env_name, args.algo, args.seed, device, args.save_dir)
    else:
        if args.model_dir is None:
            print("Error: --model_dir required for evaluation mode")
            return
        evaluate_agent(args.env_name, args.algo, args.seed, 
                      args.model_dir, N_EVAL_EPISODES)


if __name__ == "__main__":
    main()