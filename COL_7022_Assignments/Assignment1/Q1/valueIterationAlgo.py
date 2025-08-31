from env import FootballSkillsEnv
from model import model
import numpy as np

def valueIterationAlgo(envr=FootballSkillsEnv, model=model, logEnabled = True, degrade_pitch = False):
    '''
    Implements the Value Iteration algorithm to find the optimal policy for the 
    Football Skills Environment.
    
    Args:
        envr (class, optional): Environment class to instantiate. Defaults to FootballSkillsEnv.
    Returns:
        tuple: (optimal_policy, value_function, num_iterations)
            - optimal_policy (numpy.ndarray): each index maps a state to optimal actions
            - value_function (numpy.ndarray): Value of each state under optimal policy  
            - num_iterations (int): Number of iterations until convergence
    
    Algorithm:
    1. Initialize arbitrary policy and value function
    2. Value Iteration : Iteratively update value function with the maxm q-value of all actions 
    3. Repeat step 2 until value function converges
    4. Evaluate the optimal policy from the optimal value function
    '''
    model = model()
    env = envr(render_mode='gif')
    
    # get all states
    allStateTuples = model.generateAllStates(env)
    actionIndexes = model.generateAllActionIndexes(env)
    numIterations = 0
    
    # initialize policy
    policy, valueFn = model.initialize()
    # policy will say for any state, which is the right action to perform
    # valueFn will be give the sum of the accumulated rewards from step s
    
    while(True):
        numIterations += 1
        
        maxAbsDiff = 0
        for currStateTuple in allStateTuples:
                
            currStateIndex = env.state_to_index(currStateTuple)
            currentValueFn = valueFn[currStateIndex]
            
            if env._is_terminal(currStateTuple):
                continue
            
            _, valueFn[currStateIndex] = model.evaluateArgMax(currStateTuple, actionIndexes ,valueFn ,env)
            maxAbsDiff = max(maxAbsDiff, abs(valueFn[currStateIndex] - currentValueFn))
        
#         if logEnabled:
#             print("numIterations : ", numIterations, " :: maxAbsDiff :: ", maxAbsDiff)
            
        if maxAbsDiff < model.threshold:
            break
    
    # evaluate the optimal policy
    for currStateTuple in allStateTuples:
        
        currStateIndex = env.state_to_index(currStateTuple)
        policy[currStateIndex], _ = model.evaluateArgMax(currStateTuple, actionIndexes ,valueFn ,env)
    
    callsToGetTransisionsFn = len(actionIndexes)*len(allStateTuples)*(numIterations + 1)
    print("Count of total number of calls made to the  env.get_transitions_at_time is : ", callsToGetTransisionsFn)
    
    # 6
    env.get_gif(policy, seed = 20, filename = "VIOutput.gif")
    return policy, valueFn, 1, numIterations, callsToGetTransisionsFn