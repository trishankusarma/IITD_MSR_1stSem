import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv
import random
from portfolio_sol import PortFolioModel

class PolicyIterationOnPortfolioOptimization:
    def __init__(self, 
                 model,
                 epsilon=1e-6,
                 maxIterations = 1000 
                ):
        self.model = model
        self.threshold = epsilon
        pass
    
    def policy_evaluation(self, policy, valueFn):
        
        while(True):
            
            delta = 0

            # evaluate valueFn for the given policy
            # Looping it backward 
            for timeStamp in reversed(range(self.model.episodeLength+1)):
                for cashOnHold in range(self.model.maxmCash+1):
                    for assetsOnHold in range( self.model.maxmAssetLimit + 1):
                        
                        stateTuple = (timeStamp, cashOnHold, assetsOnHold)
                        # action to perform based on current policy
                        action = policy[timeStamp, cashOnHold, assetsOnHold]
                        # old value fn at current state
                        oldValueFn =  valueFn[timeStamp, cashOnHold, assetsOnHold]
                        
                        # calculate the Q_value for that (state, action)
                        valueFn[timeStamp, cashOnHold, assetsOnHold] = self.model.calculateQValue(stateTuple, action, valueFn)
                        delta = max(delta, abs(valueFn[timeStamp, cashOnHold, assetsOnHold] - oldValueFn))
            
            print(f"delta : {delta}")
            if delta < self.threshold :
                break
        return valueFn

    def policy_improvement(self, policy, valueFn):
        
        policyStable = True
        errors = 0
        
        # get all states
        for timeStamp in reversed(range(self.model.episodeLength+1)):
                for cashOnHold in range(self.model.maxmCash+1):
                    for assetsOnHold in range( self.model.maxmAssetLimit + 1):
                        
                        stateTuple = (timeStamp, cashOnHold, assetsOnHold)
                        oldPolicy = policy[timeStamp, cashOnHold, assetsOnHold]
                        
                        newPolicyForCurrState, _ = self.model.get_best_action(stateTuple, valueFn)
                        policy[timeStamp, cashOnHold, assetsOnHold] = newPolicyForCurrState
                        
                        if newPolicyForCurrState != oldPolicy:
                            policyStable = False
                            errors += 1
                            
        print(f"Errors in policy : {errors}")
        return policyStable, policy
    
    def run_policy_iteration(self, seed = 0, max_iterations=1000):  
        
        self.model.maxmCash = int(self.model.getMaxmCashPossible())
        
        print(f"Maxm price one can hold : {self.model.maxmCash}")

        # initialize policy and value function
        policy, valueFn = self.model.initializePolicyAndValueFn()
        
        print("Start evaluating policies...policy_size : ", policy.shape)
        
        num_iterations = 0
        
        while(True):
            num_iterations += 1
            
            # policy_evaluation
            valueFn = self.policy_evaluation(policy, valueFn)
            
            # policy_iteration
            policy_stable, policy = self.policy_improvement(policy, valueFn)
            
            if policy_stable == True:
                break
        
        print(f"The policy iteration took {num_iterations} iterations to converge")
        return policy, valueFn