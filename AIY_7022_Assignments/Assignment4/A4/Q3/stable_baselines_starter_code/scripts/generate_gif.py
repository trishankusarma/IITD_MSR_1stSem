"""
Generate evaluation GIFs from trained Stable Baselines3 models
Compatible with your training script structure
"""

import argparse
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import imageio
from PIL import Image, ImageDraw, ImageFont


def add_text_to_frame(frame, text, position=(10, 10), font_size=20):
    """Add text overlay to a frame"""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Add black background for text
    bbox = draw.textbbox(position, text, font=font)
    padding = 5
    draw.rectangle([bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding], fill='black')
    draw.text(position, text, fill='white', font=font)
    
    return np.array(img)


def generate_gif(env_name, algo, model_dir, output_path, n_episodes=3, fps=30, max_steps=1000):
    """
    Generate a GIF of the trained agent's performance
    
    Args:
        env_name: Name of the environment
        algo: Algorithm used ('a2c' or 'ppo')
        model_dir: Directory containing the trained model
        output_path: Path to save the GIF
        n_episodes: Number of episodes to record
        fps: Frames per second for the GIF
        max_steps: Maximum steps per episode
    """
    print(f"\nGenerating GIF for {algo.upper()} on {env_name}...")
    print(f"Model directory: {model_dir}")
    
    # Construct model path
    model_path = os.path.join(model_dir, "final_model.zip")
    vec_normalize_path = os.path.join(model_dir, "vec_normalize.pkl")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        # Try best_model.zip as fallback
        model_path = os.path.join(model_dir, "best_model.zip")
        if not os.path.exists(model_path):
            print(f"Error: No model found in {model_dir}")
            return None
        print(f"Using best_model.zip instead")
    
    # Load the model
    print(f"Loading model from: {model_path}")
    if algo == 'a2c':
        model = A2C.load(model_path)
    elif algo == 'ppo':
        model = PPO.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    # Create environment with rendering
    env = gym.make(env_name, render_mode='rgb_array')
    
    # Load normalization stats if available
    use_vec_normalize = os.path.exists(vec_normalize_path)
    
    if use_vec_normalize:
        print(f"Loading VecNormalize from: {vec_normalize_path}")
        env_wrapped = DummyVecEnv([lambda: env])
        env_wrapped = VecNormalize.load(vec_normalize_path, env_wrapped)
        env_wrapped.training = False
        env_wrapped.norm_reward = False
    
    frames = []
    episode_rewards = []
    
    print(f"\nRecording {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        if use_vec_normalize:
            obs = env_wrapped.reset()
        else:
            obs, _ = env.reset()
            obs = obs.reshape(1, -1)
        
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done and step_count < max_steps:
            # Render frame
            frame = env.render()
            
            # Add text overlay with episode info
            text = f"{algo.upper()} | Episode {episode + 1}/{n_episodes} | Step {step_count} | Reward: {episode_reward:.1f}"
            frame_with_text = add_text_to_frame(frame, text)
            frames.append(frame_with_text)
            
            # Take action
            action, _ = model.predict(obs, deterministic=True)
            
            if use_vec_normalize:
                obs, reward, done_array, info = env_wrapped.step(action)
                reward = reward[0]
                done = done_array[0]
            else:
                obs, reward, terminated, truncated, info = env.step(action[0])
                done = terminated or truncated
                obs = obs.reshape(1, -1)
            
            episode_reward += reward
            step_count += 1
        
        episode_rewards.append(episode_reward)
        print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step_count}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save GIF
    print(f"\nSaving GIF to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"GIF created successfully!")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Total frames: {len(frames)}")
    
    env.close()
    return episode_rewards


def generate_comparison_gif(env_name, a2c_model_dir, ppo_model_dir, output_path, 
                            n_episodes=1, fps=30, max_steps=1000):
    """
    Generate a side-by-side comparison GIF of A2C and PPO
    
    Args:
        env_name: Name of the environment
        a2c_model_dir: Directory containing the A2C model
        ppo_model_dir: Directory containing the PPO model
        output_path: Path to save the GIF
        n_episodes: Number of episodes to record
        fps: Frames per second for the GIF
        max_steps: Maximum steps per episode
    """
    print(f"\nGenerating comparison GIF for {env_name}...")
    
    # Load A2C model
    a2c_model_path = os.path.join(a2c_model_dir, "final_model.zip")
    if not os.path.exists(a2c_model_path):
        a2c_model_path = os.path.join(a2c_model_dir, "best_model.zip")
    print(f"Loading A2C model from: {a2c_model_path}")
    a2c_model = A2C.load(a2c_model_path)
    
    # Load PPO model
    ppo_model_path = os.path.join(ppo_model_dir, "final_model.zip")
    if not os.path.exists(ppo_model_path):
        ppo_model_path = os.path.join(ppo_model_dir, "best_model.zip")
    print(f"Loading PPO model from: {ppo_model_path}")
    ppo_model = PPO.load(ppo_model_path)
    
    # Create environments
    env_a2c = gym.make(env_name, render_mode='rgb_array')
    env_ppo = gym.make(env_name, render_mode='rgb_array')
    
    # Load normalization stats
    a2c_vec_normalize = os.path.join(a2c_model_dir, "vec_normalize.pkl")
    ppo_vec_normalize = os.path.join(ppo_model_dir, "vec_normalize.pkl")
    
    use_a2c_norm = os.path.exists(a2c_vec_normalize)
    use_ppo_norm = os.path.exists(ppo_vec_normalize)
    
    if use_a2c_norm:
        env_a2c_wrapped = DummyVecEnv([lambda: env_a2c])
        env_a2c_wrapped = VecNormalize.load(a2c_vec_normalize, env_a2c_wrapped)
        env_a2c_wrapped.training = False
        env_a2c_wrapped.norm_reward = False
    
    if use_ppo_norm:
        env_ppo_wrapped = DummyVecEnv([lambda: env_ppo])
        env_ppo_wrapped = VecNormalize.load(ppo_vec_normalize, env_ppo_wrapped)
        env_ppo_wrapped.training = False
        env_ppo_wrapped.norm_reward = False
    
    frames = []
    
    print(f"\nRecording {n_episodes} comparison episodes...")
    
    for episode in range(n_episodes):
        # Reset both environments
        if use_a2c_norm:
            obs_a2c = env_a2c_wrapped.reset()
        else:
            obs_a2c, _ = env_a2c.reset()
            obs_a2c = obs_a2c.reshape(1, -1)
        
        if use_ppo_norm:
            obs_ppo = env_ppo_wrapped.reset()
        else:
            obs_ppo, _ = env_ppo.reset()
            obs_ppo = obs_ppo.reshape(1, -1)
        
        done_a2c = False
        done_ppo = False
        reward_a2c = 0
        reward_ppo = 0
        step_count = 0
        
        while (not done_a2c or not done_ppo) and step_count < max_steps:
            # Get frames
            if not done_a2c:
                frame_a2c = env_a2c.render()
            else:
                frame_a2c = np.zeros_like(env_a2c.render())
            
            if not done_ppo:
                frame_ppo = env_ppo.render()
            else:
                frame_ppo = np.zeros_like(env_ppo.render())
            
            # Combine frames side by side
            combined_frame = np.hstack([frame_a2c, frame_ppo])
            
            # Add text overlays
            text = f"A2C: {reward_a2c:.1f} | PPO: {reward_ppo:.1f} | Step {step_count}"
            combined_frame = add_text_to_frame(combined_frame, text, font_size=25)
            frames.append(combined_frame)
            
            # Take actions
            if not done_a2c:
                action_a2c, _ = a2c_model.predict(obs_a2c, deterministic=True)
                if use_a2c_norm:
                    obs_a2c, r_a2c, done_array, _ = env_a2c_wrapped.step(action_a2c)
                    r_a2c = r_a2c[0]
                    done_a2c = done_array[0]
                else:
                    obs_a2c, r_a2c, terminated, truncated, _ = env_a2c.step(action_a2c[0])
                    done_a2c = terminated or truncated
                    obs_a2c = obs_a2c.reshape(1, -1)
                reward_a2c += r_a2c
            
            if not done_ppo:
                action_ppo, _ = ppo_model.predict(obs_ppo, deterministic=True)
                if use_ppo_norm:
                    obs_ppo, r_ppo, done_array, _ = env_ppo_wrapped.step(action_ppo)
                    r_ppo = r_ppo[0]
                    done_ppo = done_array[0]
                else:
                    obs_ppo, r_ppo, terminated, truncated, _ = env_ppo.step(action_ppo[0])
                    done_ppo = terminated or truncated
                    obs_ppo = obs_ppo.reshape(1, -1)
                reward_ppo += r_ppo
            
            step_count += 1
        
        print(f"  Episode {episode + 1}: A2C = {reward_a2c:.2f}, PPO = {reward_ppo:.2f}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save GIF
    print(f"\nSaving comparison GIF to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    print("Comparison GIF created successfully!")
    print(f"Total frames: {len(frames)}")
    
    env_a2c.close()
    env_ppo.close()


def main():
    parser = argparse.ArgumentParser(description='Generate evaluation GIFs from trained models')
    parser.add_argument('--env_name', type=str, required=True,
                        choices=['InvertedPendulum-v4', 'Hopper-v4', 'HalfCheetah-v4'],
                        help='Environment name')
    parser.add_argument('--algo', type=str, required=True,
                        choices=['a2c', 'ppo', 'comparison'],
                        help='Algorithm to evaluate or "comparison" for side-by-side')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory containing the model (for single algorithm)')
    parser.add_argument('--a2c_model_dir', type=str, default=None,
                        help='Directory containing A2C model (for comparison)')
    parser.add_argument('--ppo_model_dir', type=str, default=None,
                        help='Directory containing PPO model (for comparison)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the output GIF')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed to use for model selection')
    parser.add_argument('--n_episodes', type=int, default=3,
                        help='Number of episodes to record')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for the GIF')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps per episode')
    
    args = parser.parse_args()
    
    if args.algo == 'comparison':
        if args.a2c_model_dir is None or args.ppo_model_dir is None:
            print("Error: --a2c_model_dir and --ppo_model_dir required for comparison mode")
            return
        generate_comparison_gif(args.env_name, args.a2c_model_dir, args.ppo_model_dir,
                              args.output_path, args.n_episodes, args.fps, args.max_steps)
    else:
        if args.model_dir is None:
            print("Error: --model_dir required for single algorithm mode")
            return
        generate_gif(args.env_name, args.algo, args.model_dir,
                    args.output_path, args.n_episodes, args.fps, args.max_steps)


if __name__ == "__main__":
    main()