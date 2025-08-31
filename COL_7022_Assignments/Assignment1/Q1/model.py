import numpy as np
import pandas as pd

class model():
    def __init__(self):
        self.threshold = 1e-6
        self.discount_factor = 0.95
        self.maxStateSize = 800  # max possible states in stationary env
        self.episodeCount = 20
        self.stationary_horizon = 1
        self.non_stationary_horizon = 40
        
    def generateAllStates(self, env):
        states = []
        for x in range(env.grid_size):
            for y in range(env.grid_size):
                for is_shot_val in [0, 1]:
                    state_tuple = (x, y, is_shot_val)
                    states.append(state_tuple)
        print(f"Total number of possible states : {len(states)}")
        return states

    def generateAllActionIndexes(self, env):
        actionIndex = env.movement_actions.union(env.shooting_actions)
        print("Total actions : ", len(actionIndex))
        return actionIndex
    
    def initialize(self):
        # in stationary state, total no of states = grid_size*grid_size*2(for has_shot binary) = 800 
        # and maxm index value possible is for position (grid_size-1, grid_size-1) 
        # = ((grid_size-1) * grid_size + (grid_size-1)) * 2 + int(has_shot) = (19*20 + 19)*2 + 1 = 799
        policy = np.ones(self.maxStateSize, dtype=int)
        valueFn = np.zeros(self.maxStateSize)
               
        print(f"Shape of policy matrix {policy.shape} and shape of valueFn {valueFn.shape}")
        return policy, valueFn
    
    def getReward(self, nextState, currState, action, env):
        # since ball moves ahead of the player and the action is what he takes
        ball_pos = (nextState[0], nextState[1])
        player_pos = (currState[0], currState[1])
        reward = env._get_reward(ball_pos, action, player_pos)
        
        return reward
    
    def evaluateExpectation(self, prob, reward, valueFnOfNextState):
        return prob*(reward + self.discount_factor*valueFnOfNextState)
        
    # action here is what we get from the policy when the player obs is at currState
    def evaluateValueFunction(self, currStateTuple, policyForCurrState, valueFunction, env):
        
        # v(s) = forAll actions( policy(currStateTuple) )forall( state, reward on performing that action )P(next_state, reward)( reward + gamma*V(next_State) )
        nextStateTransisitions = env.get_transitions_at_time(currStateTuple, policyForCurrState)
        
        if not nextStateTransisitions or len(nextStateTransisitions) == 0:
            return
        
        Q_value = 0
        for prob, nextStateTuple in nextStateTransisitions:
            
            nextStateIndex = env.state_to_index(nextStateTuple)
            # get the rewards each of the nextStateTuple
            reward = self.getReward(nextStateTuple, currStateTuple, policyForCurrState, env)
            Q_value += self.evaluateExpectation(prob, reward, valueFunction[nextStateIndex])
        return Q_value
    
    def evaluateArgMax(self, currStateTuple, actionIndexes ,valueFunction , env):
        maxmQValue = -np.inf
        optimalAction = -1
        
        for action in actionIndexes:
            
            # for each action get all the next state transitions
            nextStateTransitions = env.get_transitions_at_time(currStateTuple, action)
            if not nextStateTransitions or len(nextStateTransitions) == 0:
                continue
                
            Q_value = 0
            
            for prob, nextStateTuple in nextStateTransitions:
                
                nextStateIndex = env.state_to_index(nextStateTuple)
                # get the rewards each of the next_state
                reward = self.getReward(nextStateTuple, currStateTuple, action, env)
                Q_value += self.evaluateExpectation(prob, reward, valueFunction[nextStateIndex])
                
            if Q_value > maxmQValue:
                maxmQValue = Q_value
                optimalAction = action
                
        return optimalAction, maxmQValue
    
    # Given the policy return the value function upon convergence
    def performPolicyEvaluation(self, allStateTuples, policy, valueFunction, env, logEnabled = False):
        
        numInnerIterations = 0
        
        while(True):
            numInnerIterations += 1
            maxAbsDiff = 0
            
            for currStateTuple in allStateTuples:

                if env._is_terminal(currStateTuple):
                    continue

                currStateIndex = env.state_to_index(currStateTuple)
                oldValueFunction = valueFunction[currStateIndex]
                policyForCurrState = policy[currStateIndex]

                # get the new updated valueFunction for that policy
                valueFunction[currStateIndex] = self.evaluateValueFunction(currStateTuple, policyForCurrState, valueFunction, env)
                maxAbsDiff = max(maxAbsDiff, abs(valueFunction[currStateIndex] - oldValueFunction))

            if logEnabled:
                print("numInnerIterations ",numInnerIterations," :: maxAbsDiff : ", maxAbsDiff)

            if maxAbsDiff < self.threshold:
                break
        
        return valueFunction, numInnerIterations
    
    # Given the value function update the policy upon convergence
    def performPolicyImprovement(self, allStateTuples, policy, valueFunction, actionIndexes, env):
        policyStable = True
        errors = 0
        
        for currStateTuple in allStateTuples:
            
            currStateIndex = env.state_to_index(currStateTuple)
            oldPolicyOfCurrState = policy[currStateIndex]
            policy[currStateIndex], _ = self.evaluateArgMax(currStateTuple, actionIndexes ,valueFunction ,env)
            
            if oldPolicyOfCurrState != policy[currStateIndex]:
                policyStable = False
                errors += 1
        
        print("Errors in policy_state :", errors)
        return policy, policyStable
    
    def print_policy(self, policy, grid_shape, terminal_states=[]):
        """Pretty print the football policy with text symbols"""
        symbols = {
            0: "↑",      # up
            1: "↓",      # down
            2: "←",      # left
            3: "→",      # right
            4: "S",      # short_shot_straight
            5: "LS",     # long_shot_straight
            6: "LC"      # long_shot_curl
        }

        rows, cols = grid_shape
        for r in range(rows):
            row_symbols = []
            for c in range(cols):
                state = r * cols + c
                if state in terminal_states:
                    row_symbols.append("T")
                else:
                    row_symbols.append(symbols.get(policy[state], "?"))
            print(" ".join(row_symbols))
