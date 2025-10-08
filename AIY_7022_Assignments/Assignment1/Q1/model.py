import numpy as np
import pandas as pd
import heapq

class model():
    def __init__(self):
        self.threshold = 1e-6
        self.discount_factor = 0.95
        self.maxStateSize = 800  # max possible states in stationary env
        self.episodeCount = 20
        self.non_stationary_horizon = 40
        self.predecessor_matrix = {}
    
    def setDiscountFactor(self, discount_factor):
        self.discount_factor = discount_factor
        
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
    
    def initialize(self, degrade_pitch):
        
        if degrade_pitch == False:
            # in stationary state, total no of states = grid_size*grid_size*2(for has_shot binary) = 800 
            # and maxm index value possible is for position (grid_size-1, grid_size-1) 
            # = ((grid_size-1) * grid_size + (grid_size-1)) * 2 + int(has_shot) = (19*20 + 19)*2 + 1 = 799
            policy = np.ones(self.maxStateSize, dtype=int)
            valueFn = np.zeros(self.maxStateSize)

            print(f"Shape of policy matrix {policy.shape} and shape of valueFn {valueFn.shape}")
            return policy, valueFn
        
        # in non-stationary state, the total no of states, actions remains the same but the prob and transition matrix is now dependent on time
        # here, we need to move to 2D matrix for storing the policies and value functions corresponding to each time_stamp
        # so, here we are constructing (horizon+1)*(max_state_size) matrix for both policies and value_functions
        # with base case V[40] to set to all zeros 
        policy = np.ones((self.non_stationary_horizon, self.maxStateSize), dtype=int) # policy at timestamp T does not exist
        valueFn = np.zeros((self.non_stationary_horizon+1, self.maxStateSize))

        print(f"Shape of policy matrix {policy.shape} and shape of valueFn {valueFn.shape}")
        return policy, valueFn
    
    def generatePredecessorMatrixForStationaryEnv(self, allStateTuples, actionIndexes, env):
        
        self.predecessor_matrix = np.empty(self.maxStateSize, dtype=object)
        
        for currStateTuple in allStateTuples:
            currStateIndex = env.state_to_index(currStateTuple)
            for action in actionIndexes:
                nextStateTransitions = env.get_transitions_at_time(currStateTuple, action)
                for prob, nextStateTuple in nextStateTransitions:
                    nextStateIndex = env.state_to_index(nextStateTuple)
                    
                    if not self.predecessor_matrix[nextStateIndex]:
                        self.predecessor_matrix[nextStateIndex] = []
                    
                    self.predecessor_matrix[nextStateIndex].append(currStateIndex)
    
    def getReward(self, nextState, currState, action, env):
        # since ball moves ahead of the player and the action is what he takes
        ball_pos = (nextState[0], nextState[1])
        player_pos = (currState[0], currState[1])
        reward = env._get_reward(ball_pos, action, player_pos)
        
        return reward
    
    def evaluateExpectation(self, prob, reward, valueFnOfNextState):
        return prob*(reward + self.discount_factor*valueFnOfNextState)
    
    # -------evaluateValueFunction-------#
    
    def evaluateValueFunctionForStationaryEnv(self, currStateTuple, policyForCurrState, valueFunction, env):
        # v(s) = forAll actions( policy(currStateTuple) )forall( state, reward on performing that action )P(next_state, reward)( reward + gamma*V(next_State) )
        nextStateTransisitions = env.get_transitions_at_time(currStateTuple, policyForCurrState)

        if not nextStateTransisitions or len(nextStateTransisitions) == 0:
            return 0

        Q_value = 0
        for prob, nextStateTuple in nextStateTransisitions:

            nextStateIndex = env.state_to_index(nextStateTuple)
            # get the rewards each of the nextStateTuple
            reward = self.getReward(nextStateTuple, currStateTuple, policyForCurrState, env)
            Q_value += self.evaluateExpectation(prob, reward, valueFunction[nextStateIndex])
        return Q_value
    
    def evaluateValueFunctionForNonStationaryEnv(self, currStateTuple, policyForCurrState, valueFunction, env, currTimeStamp):
        # for non-stationary environments
        # given the current-timestamp, the policy(curr_timeStamp, state), valueFunction[ timeStamp+1, state ] and the reward at that next state
        nextStateTransisitions = env.get_transitions_at_time(currStateTuple, policyForCurrState, currTimeStamp)
        
        if not nextStateTransisitions or len(nextStateTransisitions) == 0:
            return 0
        
        Q_value = 0
        for prob, nextStateTuple in nextStateTransisitions:

            nextStateIndex = env.state_to_index(nextStateTuple)
            # get the rewards each of the nextStateTuple
            reward = self.getReward(nextStateTuple, currStateTuple, policyForCurrState, env)
            Q_value += self.evaluateExpectation(prob, reward, valueFunction[currTimeStamp + 1][nextStateIndex])
        return Q_value
    
    # ----------performPolicyEvaluationForPI-------------- #
    
    # Given the policy return the value function upon convergence
    def performPolicyEvaluationForPIStationaryEnv(self, allStateTuples, policy, valueFunction, degrade_pitch, env, logEnabled):
        
        numInnerIterations = 0
        numCallsToGetTransition = 0
        
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
                valueFunction[currStateIndex] = self.evaluateValueFunction(currStateTuple, policyForCurrState, valueFunction, degrade_pitch, env)
                numCallsToGetTransition += 1
                maxAbsDiff = max(maxAbsDiff, abs(valueFunction[currStateIndex] - oldValueFunction))

            if logEnabled:
                print("numInnerIterations ",numInnerIterations," :: maxAbsDiff : ", maxAbsDiff)

            if maxAbsDiff < self.threshold:
                break
        
        return valueFunction, numCallsToGetTransition, numInnerIterations
    
    # Given the policy return the value function upon convergence
    def performPolicyEvaluationForPINonStationaryEnv(self, allStateTuples, policy, valueFunction, degrade_pitch, env, logEnabled = False):
        
        numInnerIterations = 0
        numCallsToGetTransition = 0
        
        while(True):
            numInnerIterations += 1
            maxAbsDiff = 0
            
            # in policy iteration, for finding out values for that policy, we need to iterate in reverse manner from Time t = T-1 to t = 0
            # so that we can solve it in DP manner
            for currTimeStamp in reversed(range(self.non_stationary_horizon)):
                for currStateTuple in allStateTuples:

                    if env._is_terminal(currStateTuple):
                        continue

                    currStateIndex = env.state_to_index(currStateTuple)
                    oldValueFunction = valueFunction[currTimeStamp][currStateIndex]
                    policyForCurrState = policy[currTimeStamp][currStateIndex]

                    # get the new updated valueFunction for that policy
                    valueFunction[currTimeStamp][currStateIndex] = self.evaluateValueFunction(currStateTuple, policyForCurrState, valueFunction, degrade_pitch, env, currTimeStamp)
                    numCallsToGetTransition += 1
                    maxAbsDiff = max(maxAbsDiff, abs(valueFunction[currTimeStamp][currStateIndex] - oldValueFunction))

            if logEnabled:
                print("numInnerIterations ",numInnerIterations," :: maxAbsDiff : ", maxAbsDiff)

            if maxAbsDiff < self.threshold:
                break
        
        return valueFunction, numCallsToGetTransition, numInnerIterations
    
    # ------------evaluateArgMax------------------ #
    
    def evaluateArgMaxForStationaryEnv(self, currStateTuple, actionIndexes ,valueFunction, env):
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
    
    def evaluateArgMaxForNonStationaryEnv(self, currStateTuple, actionIndexes ,valueFunction, env, currTimeStamp, passTimeStamp):
        maxmQValue = -np.inf
        optimalAction = -1
        
        for action in actionIndexes:
            
            # for each action get all the next state transitions
            if passTimeStamp == False:
                nextStateTransitions = env.get_transitions_at_time(currStateTuple, action)
            else:
                nextStateTransitions = env.get_transitions_at_time(currStateTuple, action, time_step = currTimeStamp)
            if not nextStateTransitions or len(nextStateTransitions) == 0:
                continue
                
            Q_value = 0
            
            for prob, nextStateTuple in nextStateTransitions:
                
                nextStateIndex = env.state_to_index(nextStateTuple)
                # get the rewards each of the next_state
                reward = self.getReward(nextStateTuple, currStateTuple, action, env)
                # Lookahead: use value at t+1
                if currTimeStamp + 1 < self.non_stationary_horizon:
                    Q_value += self.evaluateExpectation(prob, reward, valueFunction[currTimeStamp+1][nextStateIndex])
                else:
                    Q_value += self.evaluateExpectation(prob, reward, 0)
                
            if Q_value > maxmQValue:
                maxmQValue = Q_value
                optimalAction = action
                
        return optimalAction, maxmQValue
    
    # -----------performPolicyImprovementForPI----------------- # 
    
    # Given the value function update the policy upon convergence
    def performPolicyImprovementForPIStationaryEnv(self, allStateTuples, policy, valueFunction, actionIndexes, degrade_pitch, env):
        policyStable = True
        errors = 0
        numCallsToGetTransition = 0
        
        for currStateTuple in allStateTuples:
            
            currStateIndex = env.state_to_index(currStateTuple)
            oldPolicyOfCurrState = policy[currStateIndex]
            policy[currStateIndex], _ = self.evaluateArgMax(currStateTuple, actionIndexes ,valueFunction ,degrade_pitch, env)
            numCallsToGetTransition += len(actionIndexes)
            
            if oldPolicyOfCurrState != policy[currStateIndex]:
                policyStable = False
                errors += 1
        
        print("Errors in policy_state :", errors)
        return policy, policyStable, numCallsToGetTransition
    
    # Given the value function update the policy upon convergence
    def performPolicyImprovementForPINonStationaryEnv(self, allStateTuples, policy, valueFunction, actionIndexes, degrade_pitch, env):
        policyStable = True
        errors = 0
        numCallsToGetTransition = 0
        
        for currTimeStamp in reversed(range(self.non_stationary_horizon)):
            for currStateTuple in allStateTuples:

                currStateIndex = env.state_to_index(currStateTuple)
                oldPolicyOfCurrState = policy[currTimeStamp][currStateIndex]
                policy[currTimeStamp][currStateIndex], _ = self.evaluateArgMax(currStateTuple, actionIndexes ,valueFunction ,degrade_pitch, env, currTimeStamp)
                numCallsToGetTransition += len(actionIndexes)

                if oldPolicyOfCurrState != policy[currTimeStamp][currStateIndex]:
                    policyStable = False
                    errors += 1
        
        print("Errors in policy_state :", errors)
        return policy, policyStable, numCallsToGetTransition
    
    # ---------performValueFunctionForVI----------- #
    def performBellmanBackUp(self, currStateTuple, currStateIndex, actionIndexes, valueFn, degrade_pitch, env):
        currentValueFn = valueFn[currStateIndex]

        _, updatedFunctionVal = self.evaluateArgMax(currStateTuple, actionIndexes ,valueFn ,degrade_pitch, env)
        return updatedFunctionVal, abs(updatedFunctionVal - currentValueFn)
        
    def performValueFunctionImprovementForVIStationaryEnv(self, allStateTuples, actionIndexes, valueFn, degrade_pitch, env):
        numIterations = 0
        numCallsToGetTransition = 0
        while(True):
            numIterations += 1

            maxAbsDiff = 0
            for currStateTuple in allStateTuples:
                
                if env._is_terminal(currStateTuple):
                    continue
 
                currStateIndex = env.state_to_index(currStateTuple)
                valueFn[currStateIndex], delta = self.performBellmanBackUp(currStateTuple, currStateIndex, actionIndexes, valueFn, degrade_pitch, env)
                
                numCallsToGetTransition += len(actionIndexes)
                maxAbsDiff = max(maxAbsDiff, delta)
            
            print(f"maxAbsDiff after iteration : {numIterations} is {maxAbsDiff}")
            
            if maxAbsDiff < self.threshold:
                break
        return valueFn, numCallsToGetTransition, numIterations
    
    def performValueFunctionImprovementForVINonStationaryEnv(self, allStateTuples, actionIndexes, valueFn, degrade_pitch, passTimeStamp, env):
        numIterations = 0
        numCallsToGetTransition = 0
        
        while(True):
            numIterations += 1
            maxAbsDiff = 0
            oldValueFn = valueFn.copy()
            
            for currTimeStamp in reversed(range(self.non_stationary_horizon)):
                for currStateTuple in allStateTuples:

                    currStateIndex = env.state_to_index(currStateTuple)
                    oldValueFnAtST = oldValueFn[currTimeStamp][currStateIndex]

                    if env._is_terminal(currStateTuple):
                        continue

                    _, newValueFnAtST = self.evaluateArgMax(currStateTuple, actionIndexes ,valueFn ,degrade_pitch, env, currTimeStamp, passTimeStamp)
                    
                    numCallsToGetTransition += len(actionIndexes)
                    oldValueFn[currTimeStamp][currStateIndex] = newValueFnAtST
                    maxAbsDiff = max(maxAbsDiff, abs(newValueFnAtST - oldValueFnAtST))

            print(f"maxAbsDiff after iteration : {numIterations} is {maxAbsDiff}")
            valueFn  = oldValueFn
            if maxAbsDiff < self.threshold:
                break
        return valueFn, numCallsToGetTransition, numIterations
    
    # ---------performPolicyImprovementForVI----------- # 
    def performPolicyImprovementForVIStationaryEnv(self, allStateTuples, actionIndexes, valueFn, policy, degrade_pitch, env):
        # evaluate the optimal policy
        for currStateTuple in allStateTuples:
            
            currStateIndex = env.state_to_index(currStateTuple)
            policy[currStateIndex], _ = self.evaluateArgMax(currStateTuple, actionIndexes , valueFn, degrade_pitch, env)
        return policy
    
    def performPolicyImprovementForVINonStationaryEnv(self, allStateTuples, actionIndexes, valueFn, policy, degrade_pitch, passTimeStamp, env):
        # evaluate the optimal policy
        for currTimeStamp in range(self.non_stationary_horizon):
            
            for currStateTuple in allStateTuples:
                currStateIndex = env.state_to_index(currStateTuple)
                policy[currTimeStamp][currStateIndex], _ = self.evaluateArgMax(currStateTuple, actionIndexes , valueFn, degrade_pitch, env, currTimeStamp, passTimeStamp)
        return policy
            
    # --------------------------------------------Main functions below--------------------------------------------------------------------------- #
    
    # action here is what we get from the policy when the player obs is at currState
    def evaluateValueFunction(self, currStateTuple, policyForCurrState, valueFunction, degrade_pitch, env, currTimeStamp = None):
        
        if degrade_pitch == False:
            return self.evaluateValueFunctionForStationaryEnv(currStateTuple, policyForCurrState, valueFunction, env)
        return self.evaluateValueFunctionForNonStationaryEnv(currStateTuple, policyForCurrState, valueFunction, env, currTimeStamp)
    
    # Given the policy return the value function upon convergence
    def performPolicyEvaluationForPI(self, allStateTuples, policy, valueFunction, degrade_pitch, env, logEnabled = False):
        
        if degrade_pitch == False:
            return self.performPolicyEvaluationForPIStationaryEnv(allStateTuples, policy, valueFunction, degrade_pitch, env, logEnabled)
        return self.performPolicyEvaluationForPINonStationaryEnv(allStateTuples, policy, valueFunction, degrade_pitch, env, logEnabled)
    
    def evaluateArgMax(self, currStateTuple, actionIndexes ,valueFunction , degrade_pitch, env, currTimeStamp = None, passTimeStamp = True):
        
        if degrade_pitch == False:
            return self.evaluateArgMaxForStationaryEnv(currStateTuple, actionIndexes ,valueFunction, env)
        return self.evaluateArgMaxForNonStationaryEnv(currStateTuple, actionIndexes ,valueFunction, env, currTimeStamp, passTimeStamp)
    
    def performPolicyImprovementForPI(self, allStateTuples, policy, valueFunction, actionIndexes, degrade_pitch, env):
        
        if degrade_pitch == False:
            return self.performPolicyImprovementForPIStationaryEnv(allStateTuples, policy, valueFunction, actionIndexes, degrade_pitch, env)
        return self.performPolicyImprovementForPINonStationaryEnv(allStateTuples, policy, valueFunction, actionIndexes, degrade_pitch, env)
    
    def performValueFunctionImprovementForVI(self, allStateTuples, actionIndexes, valueFn, degrade_pitch, passTimeStamp, env):
        
        if degrade_pitch == False:
            return self.performValueFunctionImprovementForVIStationaryEnv(allStateTuples, actionIndexes, valueFn, degrade_pitch, env)
        return self.performValueFunctionImprovementForVINonStationaryEnv(allStateTuples, actionIndexes, valueFn, degrade_pitch, passTimeStamp, env)
    
    def performPolicyImprovementForVI(self, allStateTuples, actionIndexes, valueFn, policy, degrade_pitch, passTimeStamp, env):
        
        if degrade_pitch == False:
            return self.performPolicyImprovementForVIStationaryEnv(allStateTuples, actionIndexes, valueFn, policy, degrade_pitch, env)
        return self.performPolicyImprovementForVINonStationaryEnv(allStateTuples, actionIndexes, valueFn, policy, degrade_pitch, passTimeStamp, env)
    
    # This is to be implement only for the stationary environment
    def performModifiedValueFunctionEvaluation(self,allStateTuples, actionIndexes, valueFn, env):
        degrade_pitch = False
        numCallsToGetTransition = 0
        numIterations = 0
        # First create a priority queue, do one round of belmann value funtion evaluation(maxm one) push the states with the maxDiff based on the current initialization
        # Priority queue (negative priority because heapq is min-heap)
        pq = []
        for currStateTuple in allStateTuples:
            
            if env._is_terminal(currStateTuple):
                continue

            # Updating inline here will allow me to do one round of sweep to the goal state 
            currStateIndex = env.state_to_index(currStateTuple)
            updatedValueFnForCurrState, currDelta = self.performBellmanBackUp(currStateTuple, currStateIndex, actionIndexes, valueFn, degrade_pitch, env)
            valueFn[currStateIndex] = updatedValueFnForCurrState  # <-- one initial sweep update
            numCallsToGetTransition += len(actionIndexes)
            
            if currDelta >= self.threshold:
                heapq.heappush(pq, (-currDelta, currStateTuple))   
                # state in the priority queue = ( -delta, currStateTuple ) -> top element will be the one with maxm delta
        
        # Next step is to iterate the queue
        while pq:
            delta, currStateTuple = heapq.heappop(pq)
            delta *= (-1)
            numIterations += 1
            
            if env._is_terminal(currStateTuple):
                continue
            
            # step 3: perform bellman back up and update valueFn[currStateIndex]
            currStateIndex = env.state_to_index(currStateTuple)
            
            updatedValueFnForCurrState, updatedDelta = self.performBellmanBackUp(currStateTuple, currStateIndex, actionIndexes, valueFn, degrade_pitch, env)
            valueFn[currStateIndex] = updatedValueFnForCurrState
            numCallsToGetTransition += len(actionIndexes)
    
            if updatedDelta > self.threshold:
                # step 4: Update predecessors if significant change
                for prevStateIndex in self.predecessor_matrix[currStateIndex]:
                    prevStateTuple = env.index_to_state(prevStateIndex)
                    
                    if env._is_terminal(prevStateTuple):
                        continue
                    heapq.heappush(pq, (-updatedDelta, prevStateTuple))                         
            
            if numIterations%1000 == 0:
                print(f"Curr iteration : {numCallsToGetTransition}")
        
        return valueFn, numCallsToGetTransition, numIterations
    
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
