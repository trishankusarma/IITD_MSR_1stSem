import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import imageio
import numpy as np
import math
import os
import random
import pandas as pd 
from dataclasses import dataclass
from collections import defaultdict
from frozenlake import DiagonalFrozenLake
from cliff import MultiGoalCliffWalkingEnv
from utils.algoUtils import run_algorithm, SARSA_UPDATE_RULE, Q_UPDATE_RULE, EXPECTED_SARSA_UPDATE_RULE

def SARSA(env, **kwargs):
    '''
    Implement the SARSA algorithm to find the optimal policy for the given environment.
    return: best Q table -> np.array of shape (num_states, num_actions)
    return average training_rewards across 10 seeds-> []
    return: average safe_visits across 10 seeds -> float
    return: average risky_visits across 10 seeds -> float
    '''
    return run_algorithm(env, SARSA_UPDATE_RULE, **kwargs)

def q_learning_for_cliff(env, **kwargs):
    '''
    Implement the Q-learning algorithm to find the optimal policy for the given environment.
    return: best Q table -> np.array of shape (num_states, num_actions)
    return average training_rewards across 10 seeds-> []
    return: average safe_visits across 10 seeds -> float
    return: average risky_visits across 10 seeds -> float
    '''
    return run_algorithm(env, Q_UPDATE_RULE, **kwargs)

def expected_SARSA(env, **kwargs):
    '''
    Implement the Expected SARSA algorithm to find the optimal policy for the given environment.
    return: best Q table -> np.array of shape (num_states, num_actions)
    return average training_rewards across 10 seeds-> []
    return: average safe_visits across 10 seeds -> float
    return: average risky_visits across 10 seeds -> float
    '''
    return run_algorithm(env, EXPECTED_SARSA_UPDATE_RULE, **kwargs)


def monte_carlo(env):
    '''
    Implement the Monte Carlo algorithm to find the optimal policy for the given environment.
    Return Q table.
    return: Q table -> np.array of shape (num_states, num_actions)
    return: episode_rewards -> []
    return: _ 
    return: _ 
    '''
    pass

    return Q, episode_rewards, _, _

def q_learning_for_frozenlake(env):
    '''
    Implement the Q-learning algorithm to find the optimal policy for the given environment.
    return: Q table -> np.array of shape (num_states, num_actions)
    return episode_rewards_for_one_seed -> []
    '''
    pass

    return Q, episode_rewards, _, _