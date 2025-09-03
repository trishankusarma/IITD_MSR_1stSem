import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv
import random

class PolicyIterationOnlineKnapsack:
    def __init__(self, env, gamma=0.95, epsilon=1e-4, episodeLength = 50, maxmKnapsackWeight = 200, numItems = 200, maxIterations = 1000):
        self.env = env
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
        # returns reward, done, next_weight
        
        if action == 0:
            return 0, False, current_weight
        
        weight = self.env.item_weights[item_idx]
        value = self.env.item_values[item_idx]
        
        if( (weight + current_weight) > self.maxmKnapsackWeight ):
            # this would ensure that he don't choose this action
            return -1e9, True, current_weight
        
        return value, False, current_weight + weight
    
    def evaluateQValuesForBothAction(self, valueFn, stateTuple, action, expectedQValueFromFutureStates = None):
        
        timeRemaining, usedCapacity, itemIdx = stateTuple
        
        if expectedQValueFromFutureStates == None:
            expectedQValueFromFutureStates = self.discount_factor * np.dot( self.env.item_probs,valueFn[timeRemaining-1, usedCapacity, :])
        
        if action == 0:
            return expectedQValueFromFutureStates
    
        reward, done, updatedUsedCapacity = self.get_reward_and_done(usedCapacity, itemIdx, 1)
        
        return reward +  self.discount_factor * np.dot(
            self.env.item_probs,
            valueFn[timeRemaining-1, updatedUsedCapacity, :]
        )
    
    def policy_evaluation(self, policy, valueFn):
        
        while(True):
            
            delta = 0

            # evaluate valueFn for the given policy
            # looping backward from timeRemaining 1 to timeRemaining 50
            # and at timeRemaining 0 -> it is a terminal state -> so value should be 0
            for timeRemaining in range(1, self.episodeLength+1):
                # usedCapacity = 200 -> we can only do one action -> reject the item
                for usedCapacity in range(self.maxmKnapsackWeight+1):
                    
                    expectedQValueFromFutureStates = self.discount_factor * np.dot( self.env.item_probs,valueFn[timeRemaining-1, usedCapacity, :])
                    for itemIdx in range(self.numItems):

                        oldValueFn = valueFn[timeRemaining, usedCapacity, itemIdx]
                        stateTuple = ( timeRemaining, usedCapacity, itemIdx )

                        action = policy[timeRemaining, usedCapacity, itemIdx]

                        Q_value = self.evaluateQValuesForBothAction(valueFn, stateTuple, action, expectedQValueFromFutureStates = expectedQValueFromFutureStates)

                        valueFn[timeRemaining][usedCapacity][itemIdx] = Q_value
                        delta = max(delta, abs(Q_value - oldValueFn))

            print(f"delta : {delta}")
            
            if delta < self.threshold :
                break
        return valueFn

    def policy_improvement(self, policy, valueFn):
        policyStable = True
        errors = 0
        
        # get all states
        for timeRemaining in range(1, self.episodeLength+1):
                for usedCapacity in range(self.maxmKnapsackWeight + 1):
                    
                    expectedQValueFromFutureStates = self.discount_factor * np.dot( self.env.item_probs,valueFn[timeRemaining-1, usedCapacity, :])
                    for itemIdx in range(self.numItems):
                        
                        oldPolicy = policy[timeRemaining][usedCapacity][itemIdx]
                        stateTuple = ( timeRemaining, usedCapacity, itemIdx )
                        
                        newPolicyForCurrState = self.get_action(stateTuple, valueFn, expectedQValueFromFutureStates = expectedQValueFromFutureStates)
                        
                        policy[timeRemaining][usedCapacity][itemIdx] = newPolicyForCurrState
                        if newPolicyForCurrState != oldPolicy:
                            policyStable = False
                            errors += 1
        
        print(f"Errors in policy : {errors}")
        return policyStable, policy

    def get_action(self, stateTuple, valueFn, expectedQValueFromFutureStates = None):
        timeRemaining, usedCapacity, itemIdx = stateTuple
        
        reward, _, _ = self.get_reward_and_done(usedCapacity, itemIdx, 1)
        
        # this handles the condition for out-bounds
        if(reward < -1e8):
            return 0
        
        if expectedQValueFromFutureStates == None:
            expectedQValueFromFutureStates = self.discount_factor * np.dot( self.env.item_probs,valueFn[timeRemaining-1, usedCapacity, :])
        
        Q_reject = self.evaluateQValuesForBothAction(valueFn, stateTuple, 0, expectedQValueFromFutureStates = expectedQValueFromFutureStates)
        Q_accept = self.evaluateQValuesForBothAction(valueFn, stateTuple, 1)
                        
        if Q_reject >= Q_accept:
            return 0
        return 1
    
    def run_policy_iteration(self, seed = 0, max_iterations=1000):   

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
