import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv

class PolicyIterationOnlineKnapsack:
    def __init__(self, env=OnlineKnapsackEnv, gamma=0.95, epsilon=1e-4, episodeLength = 50, maxmKnapsackWeight = 200, numItems = 200, maxIterations = 1000):
        self.env = OnlineKnapsackEnv()
        self.discount_factor = gamma
        self.threshold = epsilon
        self.episodeLength = episodeLength
        self.maxmKnapsackWeight = maxmKnapsackWeight
        self.numItems = numItems
        self.maxIterations = maxIterations
        pass
    
    def initializePolicyAndValueFn(self):
        policy =  np.zeros((self.episodeLength+1, self.maxmKnapsackWeight+1, self.numItems), dtype=np.int8)
        valueFn = np.zeros((self.episodeLength+1, self.maxmKnapsackWeight+1, self.numItems))
        
        return policy, valueFn 

    def get_reward_and_done(self, current_weight, item_idx, action):
        
        if action == 0:
            return 0, False, current_weight
        
        weight = self.env.item_weights[item_idx]
        value = self.env.item_values[item_idx]
        
        if( (weight + current_weight) > self.maxmKnapsackWeight ):
            return float('-inf'), True, current_weight
        
        return value, False, current_weight + weight
    
    def evaluateQValuesForBothAction(self, valueFn, stateTuple, action):
        
        timeRemaining, usedCapacity, itemIdx = stateTuple
        
        if action == 0:
            return self.discount_factor * np.dot(
                self.env.item_probs,
                valueFn[timeRemaining-1, usedCapacity, :]
            )
    
        reward, done, updatedUsedCapacity = self.get_reward_and_done(usedCapacity, itemIdx, 1)
        
        return reward +  self.discount_factor * np.dot(
            self.env.item_probs,
            valueFn[timeRemaining-1, updatedUsedCapacity, :]
        )
    
    def policy_evaluation(self, policy, valueFn):
        
        delta = 0
        
        # evaluate valueFn for the given policy
        for timeRemaining in range(self.episodeLength):
            for usedCapacity in reversed(range(self.maxmKnapsackWeight)):
                for itemIdx in range(self.numItems):
                        
                    oldValueFn = valueFn[timeRemaining][usedCapacity][itemIdx]
                    stateTuple = ( timeRemaining, usedCapacity, itemIdx )
                        
                    action = policy[timeRemaining][usedCapacity][itemIdx]
                        
                    Q_value = self.evaluateQValuesForBothAction(valueFn, stateTuple, action)
            
                    valueFn[timeRemaining][usedCapacity][itemIdx] = Q_value
                    delta = max(delta, abs(Q_value - oldValueFn))
            
        print(f"delta : {delta}")
        return valueFn

    def policy_improvement(self, policy, valueFn):
        policyStable = True
        errors = 0
        
        # get all states
        for timeRemaining in range(self.episodeLength):
                for usedCapacity in reversed(range(self.maxmKnapsackWeight)):
                    for itemIdx in range(self.numItems):
                        
                        oldPolicy = policy[timeRemaining][usedCapacity][itemIdx]
                        stateTuple = ( timeRemaining, usedCapacity, itemIdx )
                        
                        newPolicyForCurrState = self.get_action(stateTuple, valueFn)
                        
                        policy[timeRemaining][usedCapacity][itemIdx] = newPolicyForCurrState
                        if newPolicyForCurrState != oldPolicy:
                            policyStable = False
                            errors += 1
        
        print(f"Errors in policy : {errors}")
        return policyStable, policy

    def get_action(self, stateTuple, valueFn):
        Q_reject = self.evaluateQValuesForBothAction(valueFn, stateTuple, 0)
        Q_accept = self.evaluateQValuesForBothAction(valueFn, stateTuple, 1)
                        
        if Q_reject > Q_accept:
            return 0
        return 1
    
    def run_policy_iteration(self, max_iterations=1000):   

        print(f"Shape of item_values : {self.env.item_values.shape}")
        
        policy, valueFn = self.initializePolicyAndValueFn()
        print(f"Shape of policy : {policy.shape}")
        print(f"Shape of valueFn : {valueFn.shape}")
        
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
