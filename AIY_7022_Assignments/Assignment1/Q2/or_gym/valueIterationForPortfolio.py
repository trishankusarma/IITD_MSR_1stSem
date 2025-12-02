import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv
import random

class ValueIterationOnPortfolioOptimization:
    def __init__(self, 
                 model,
                 epsilon=1e-6,
                 maxIterations = 1000 
                ):
        self.model = model
        self.threshold = epsilon
        pass
    
    def policy_evaluation(self, valueFn):
        
        num_iteration = 0
        while(True):
            
            delta = 0

            # evaluate valueFn for the given policy
            # Looping it backward 
            for timeStamp in reversed(range(self.model.episodeLength+1)):
                for cashOnHold in range(self.model.maxmCash+1):
                    for assetsOnHold in range( self.model.maxmAssetLimit + 1):
                        
                        stateTuple = (timeStamp, cashOnHold, assetsOnHold)        
                        # old value fn at current state
                        oldValueFn =  valueFn[timeStamp, cashOnHold, assetsOnHold]
                        
                        # calculate the maxm Q_value for that (state)
                        _, maxmQValue = self.model.get_best_action(stateTuple, valueFn)
                        valueFn[timeStamp, cashOnHold, assetsOnHold] = maxmQValue
                        delta = max(delta, abs(maxmQValue - oldValueFn))
            
            num_iteration += 1
            print(f"delta : {delta}")
            if delta < self.threshold :
                break
        return valueFn, num_iteration

    def policy_improvement(self, policy, valueFn):
        
        # get all states
        for timeStamp in reversed(range(self.model.episodeLength+1)):
                for cashOnHold in range(self.model.maxmCash+1):
                    for assetsOnHold in range( self.model.maxmAssetLimit + 1):
                        
                        stateTuple = (timeStamp, cashOnHold, assetsOnHold)
                        oldPolicy = policy[timeStamp, cashOnHold, assetsOnHold]
                        
                        newPolicyForCurrState, _ = self.model.get_best_action(stateTuple, valueFn)
                        policy[timeStamp, cashOnHold, assetsOnHold] = newPolicyForCurrState
                    
        return policy
    
    def run_value_iteration(self, seed = 0, max_iterations=1000):  
        
        self.model.maxmCash = int(self.model.getMaxmCashPossible())
        
        print(f"Maxm price one can hold : {self.model.maxmCash}")

        # initialize policy and value function
        policy, valueFn = self.model.initializePolicyAndValueFn()
        
        print("Start evaluating policies...policy_size : ", policy.shape)
        
        valueFn, num_iterations = self.policy_evaluation(valueFn)
        policy = self.policy_improvement(policy, valueFn)
        
        print(f"The policy iteration took {num_iterations} iterations to converge")
        return policy, valueFn