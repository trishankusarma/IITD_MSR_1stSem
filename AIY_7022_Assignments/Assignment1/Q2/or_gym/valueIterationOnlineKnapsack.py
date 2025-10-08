import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv

class ValueIterationOnlineKnapsack:
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
            return -1e9, True, current_weight
        
        return value, False, current_weight + weight

    def get_action(self, stateTuple, valueFn, expectedQValueFromFutureStates = None):
        
        timeRemaining, usedCapacity, itemIdx = stateTuple
        
        reward, _, _ = self.get_reward_and_done(usedCapacity, itemIdx, 1)
        
        if(reward  < -1e8):
            return 0
        
        Q_reject, Q_accept = self.evaluateQValuesForBothAction(valueFn, stateTuple)
                        
        if Q_reject >= Q_accept:
            return 0
        return 1
    
    def evaluateQValuesForBothAction(self, valueFn, stateTuple, expectedQValueFromFutureStates = None):
        
        timeRemaining, usedCapacity, itemIdx = stateTuple
        
        if expectedQValueFromFutureStates == None:
            expectedQValueFromFutureStates = self.discount_factor * np.dot( self.env.item_probs,valueFn[timeRemaining-1, usedCapacity, :])
        
        Q_reject = expectedQValueFromFutureStates
    
        reward, done, updatedUsedCapacity = self.get_reward_and_done(usedCapacity, itemIdx, 1)
        
        Q_accept = reward +  self.discount_factor * np.dot(
            self.env.item_probs,
            valueFn[timeRemaining-1, updatedUsedCapacity, :]
        )
        return Q_reject, Q_accept
    
    def policy_evaluation(self, valueFn):
        
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

                        oldValueFn = valueFn[timeRemaining][usedCapacity][itemIdx]
                        stateTuple = ( timeRemaining, usedCapacity, itemIdx )

                        Q_reject, Q_accept = self.evaluateQValuesForBothAction(valueFn, stateTuple, expectedQValueFromFutureStates = expectedQValueFromFutureStates)

                        updatedValueFn  = max(Q_reject, Q_accept)
                        valueFn[timeRemaining][usedCapacity][itemIdx] = updatedValueFn
                        delta = max(delta, abs(oldValueFn - updatedValueFn))

            print(f"delta : {delta}")
            
            if delta < self.threshold:
                break
        return valueFn
    
    def policy_improvement(self, valueFn, policy):
        
        # get all states
        # evaluate valueFn for the given policy
        # looping backward from timeRemaining 1 to timeRemaining 50
        # and at timeRemaining 0 -> it is a terminal state -> so value should be 0
        for timeRemaining in range(1, self.episodeLength+1):
            # usedCapacity = 200 -> we can only do one action -> reject the item
            for usedCapacity in range(self.maxmKnapsackWeight+1):
                    
                expectedQValueFromFutureStates = self.discount_factor * np.dot( self.env.item_probs,valueFn[timeRemaining-1, usedCapacity, :])
                for itemIdx in range(self.numItems):
                        
                    stateTuple = ( timeRemaining, usedCapacity, itemIdx ) 
                    newPolicyForCurrState = self.get_action(stateTuple, valueFn, expectedQValueFromFutureStates = expectedQValueFromFutureStates)
                        
                    policy[timeRemaining][usedCapacity][itemIdx] = newPolicyForCurrState
                        
        return policy
    
    def run_value_iteration(self, max_iterations=1000):   

        print(f"Shape of item_values : {self.env.item_values.shape}")
        
        policy, valueFn = self.initializePolicyAndValueFn()
        print(f"Shape of policy : {policy.shape}")
        print(f"Shape of valueFn : {valueFn.shape}")
        
        # policy_evaluation
        valueFn = self.policy_evaluation(valueFn)
            
        # policy_improvement
        policy = self.policy_improvement(valueFn, policy)
        
        return policy, valueFn
    