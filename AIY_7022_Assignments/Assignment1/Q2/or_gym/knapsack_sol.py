import numpy as np
from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv
import matplotlib.pyplot as plt
from scipy.stats import mode

# development I am doing in kernel notebook 
# this file is not getting used
if __name__=="__main__":
    env=OnlineKnapsackEnv()
    state=env._RESET()
    print(f"State for OnlineKnapsackEnv {state}")

    total_reward = 0
    done = False


    ###################### Random sampling #####################################################

    print("Starting Online Knapsack Simulation")
    print("Initial state:", state)

    # Run simulation until episode ends
    while not done:
        # Choose an action randomly: 0 (reject) or 1 (accept)
        action = env.sample_action()
        print(action)
        # Take a step in the environment
        next_state, reward, done, info = env._STEP(action)
        
        total_reward += reward
        print(f"Action: {'Accept' if action == 1 else 'Reject'} | Reward: {reward} | Next state: {next_state} | Done: {done}")
        exit()
    print(f"Episode finished. Total reward: {total_reward}")

