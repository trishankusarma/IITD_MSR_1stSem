import numpy as np
import itertools
from collections import deque
import matplotlib.pyplot as plt
from scipy.stats import mode
import sys
import copy
import time
import random

from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv

random.seed(42)       # Set seed for Python's built-in random module
np.random.seed(42) 

class PortFolioModel:
    
    def __init__(self, env, gamma, episodeLength = 10):
        self.env = env
        self.discount_factor = gamma
        self.episodeLength = episodeLength
        self.maxmCash = 0
        self.maxmAssetLimit = self.env.holding_limit[0] #since we have to deal with only one item
        self.txnCost = 1
        self.prices = self.env.asset_prices[0]
        self.initial_cash = self.env.initial_cash
        self.penalty = -1e9 # for handling out of bounds
        pass
        
    def initializePolicyAndValueFn(self):
        
        # action maps to 0 to 4 th index of self.actions
        # expecting to hold all policy at start
        policy =  np.full((self.episodeLength+1, self.maxmCash+1, self.maxmAssetLimit + 1), 2, dtype=np.int8)
        valueFn = np.zeros((self.episodeLength+1, self.maxmCash+1, self.maxmAssetLimit + 1))
        
        return policy, valueFn
    
    def getMaxmCashPossible(self, maxCash = 1000):
        # cashInHold = initial_cash + (horizon - ceil(maxmAssetLimit/2) - 1)*(maxm - minm) - (maxmAssetLimit)*minm - (horizon-1)*txnCost
        # assetWealth = maxmAssetLimit*maxm        
        maxPrice = max(self.prices)
        minPrice = min(self.prices)
        
        assetWealth = self.maxmAssetLimit*maxPrice
        
        expectedMaxCash = self.initial_cash + (self.episodeLength - np.ceil(self.maxmAssetLimit/2) - 1)*(maxPrice - minPrice) - self.maxmAssetLimit*minPrice - (self.episodeLength-1)*self.txnCost + assetWealth
        
        return max(expectedMaxCash, maxCash)
    
    def evaluateExpectation(self, stateTuple, k, valueFn, type = None):
        
        timeStamp, cashOnHold, assetsOnHold = stateTuple
       
        # during buying time
        if type == 1:
            
            newCash = cashOnHold - k*(self.prices[timeStamp] + self.txnCost)
            # one should have 2*price[timeStamp] + self.txnCost <= cashOnHold and (assetsOnHold + 2) <= self.maxmAssetLimit
            if( newCash < 0  or ((assetsOnHold + k) > self.maxmAssetLimit)):
                return self.penalty, (-1, -1, -1), 1
            return self.discount_factor*valueFn[timeStamp+1, newCash, assetsOnHold + k], (timeStamp+1, newCash, assetsOnHold + k), 0
        
        # during selling time
        newCash = cashOnHold+k*(self.prices[timeStamp]-self.txnCost)
        if( assetsOnHold < k or newCash > self.maxmCash):
            return self.penalty, (-1, -1, -1), 1
        
        return self.discount_factor*valueFn[ timeStamp+1, newCash, assetsOnHold-k ], (timeStamp+1, newCash, assetsOnHold-k), 0
    
    def calculateQValue(self, stateTuple, action, valueFn):
        
        timeStamp, cashOnHold, assetsOnHold = stateTuple
        
        # get reward only at the final step
        if( timeStamp == (self.episodeLength) ):
            return cashOnHold + self.prices[-1]*assetsOnHold
        
        if cashOnHold == 0:
            return 0
        
        if action == 0:
            # buy 2
            qValue, _, _ = self.evaluateExpectation(stateTuple, 2, valueFn, type = 1)
            
        elif action == 1:
            # biy 1
            qValue, _, _= self.evaluateExpectation(stateTuple, 1, valueFn, type = 1)
            
        elif action == 2:
            # hold
            qValue = self.discount_factor*valueFn[timeStamp+1, cashOnHold, assetsOnHold]
            
        elif action == 3:
            # sell 1
            qValue, _, _= self.evaluateExpectation(stateTuple, 1, valueFn, type = -1)
        else:
            # sell 2
            # if assetsOnHold  >= 2
            # return to ( timeStamp+1, cashOnHold+2*price[timeStamp], assetsOnHold-2)
            qValue, _, _ = self.evaluateExpectation(stateTuple, 2, valueFn, type = -1)
        
        return qValue
    
    def get_best_action(self, stateTuple, valueFn):
        
        maxQValue = -1e9
        best_action = 2 # default hold
        
        for action in range(5):
            curr_q_value = self.calculateQValue(stateTuple, action, valueFn)
            
            if(curr_q_value > maxQValue ):
                maxQValue = curr_q_value
                best_action = action
            
        return best_action, maxQValue
    
    def get_next_action_reward_and_done(self, stateTuple, action, valueFn):
        timeStamp, cashOnHold, assetsOnHold = stateTuple
        
        # get reward only at the final step
        if( timeStamp == (self.episodeLength) ):
            return stateTuple, cashOnHold + self.prices[-1]*assetsOnHold, 1
        
        if cashOnHold == 0:
            return stateTuple, self.penalty, 1
        
        if action == 0:
            # buy 2
            _, nextState, done = self.evaluateExpectation(stateTuple, 2, valueFn, type = 1)
            
        elif action == 1:
            # biy 1
            _, nextState, done = self.evaluateExpectation(stateTuple, 1, valueFn, type = 1)
            
        elif action == 2:
            # hold
            nextState, done = ( timeStamp+1, cashOnHold, assetsOnHold ), 0
            
        elif action == 3:
            # sell 1
            _, nextState, done = self.evaluateExpectation(stateTuple, 1, valueFn, type = -1)
            
        else:
            # sell 2
            # if assetsOnHold  >= 2
            # return to ( timeStamp+1, cashOnHold+2*price[timeStamp], assetsOnHold-2)
            _, nextState, done = self.evaluateExpectation(stateTuple, 2, valueFn, type = -1)
        
        return nextState, 0, done
    
    # update inside your PortFolioModel class
    def getTotalRewards(self, policy, valueFn, startState, method):
        totalRewards = 0
        currState = startState

        mapForAction = ["Buy 2", "Buy 1", "Hold", "Sell 1", "Sell 2"]

        # store trajectories
        trajectory_cash = []
        trajectory_holdings = []
        trajectory_wealth = []

        for i in range(self.episodeLength+1):
            currTimeStamp, currCash, currAssets = currState

            # record states
            trajectory_cash.append(currCash)
            trajectory_holdings.append(currAssets)
            trajectory_wealth.append(currCash + currAssets * self.prices[min(currTimeStamp, len(self.prices)-1)])

            if i < self.episodeLength:
                action = policy[currTimeStamp, currCash, currAssets]
                print(f"At state : {currState}, best action u can take is {mapForAction[action]}")

                nextState, reward, done = self.get_next_action_reward_and_done(currState, action, valueFn)
                currState = nextState
                totalRewards += reward

        print(f"[{method}] Final Wealth: {trajectory_wealth[-1]}")
        print(f"[{method}] Total Reward: {totalRewards}")

        # call plotting function
        self.plot_trajectories(trajectory_wealth, trajectory_cash, trajectory_holdings, method)

        return totalRewards
    
    def plot_trajectories(self, wealth, cash, holdings, method="VI"):
        plt.figure(figsize=(10,6))
        plt.plot(wealth, label="Wealth", marker="o")
        plt.plot(cash, label="Cash", marker="s")
        plt.plot(holdings, label="Holdings", marker="^")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title(f"Portfolio Evolution ({method})")
        plt.legend()
        plt.grid(True)
        plt.show()


# development I am doing in kernel notebook :: will add here once completed
if __name__=="__main__":
    start_time=time.time()


    ###Part 1 and Part 2
    ####Please train the value and policy iteration training algo for the given three sequences of prices
    ####Config1
    env = DiscretePortfolioOptEnv(prices=[1, 3, 5, 5 , 4, 3, 2, 3, 5, 8])

    ####Config2
    env = DiscretePortfolioOptEnv(prices=[2, 2, 2, 4 ,2, 2, 4, 2, 2, 2])

    ####Config3
    env = DiscretePortfolioOptEnv(prices=[4, 1, 4, 1 ,4, 4, 4, 1, 1, 4])



    ####Run the evaluation on the following prices and save the plots.


    ###Part 3. (Portfolio Optimizaton)
    env = DiscretePortfolioOptEnv(variance=1)
