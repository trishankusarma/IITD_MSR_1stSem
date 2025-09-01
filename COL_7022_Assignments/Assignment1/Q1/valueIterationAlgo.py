from env import FootballSkillsEnv
from model import model
import numpy as np

def valueIterationAlgo(envr=FootballSkillsEnv, model=model, logEnabled = True, degrade_pitch = False, passTimeStamp = True, discount_factor = 0.95, modified_VI =  False):
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
    model.setDiscountFactor(discount_factor)
    
    if degrade_pitch == False:
        env = envr(render_mode='gif')
    else:
        env = envr(render_mode='gif', degrade_pitch=True)
    
    # get all states
    allStateTuples = model.generateAllStates(env)
    actionIndexes = model.generateAllActionIndexes(env)
    
    # initialize policy
    policy, valueFn = model.initialize(degrade_pitch)
    # policy will say for any state, which is the right action to perform
    # valueFn will be give the sum of the accumulated rewards from step s
    
    if modified_VI == True:
        model.generatePredecessorMatrixForStationaryEnv(allStateTuples, actionIndexes, env)
    
    # Step 1
    if modified_VI == False:
        valueFn, numCallsToGetTransition, numIterations = model.performValueFunctionImprovementForVI(allStateTuples, actionIndexes, valueFn, degrade_pitch, passTimeStamp, env)
    else :
        valueFn, numCallsToGetTransition, numIterations = model.performModifiedValueFunctionEvaluation(allStateTuples, actionIndexes, valueFn, env)
    
    # Step 2 :: Final one
    policy = model.performPolicyImprovementForVI(allStateTuples, actionIndexes, valueFn, policy, degrade_pitch, passTimeStamp, env)
    numCallsToGetTransition += len(allStateTuples)*len(actionIndexes)
    
    # 6
#     if degrade_pitch == False:
        
#         if modified_VI == False:
#             env.get_gif(policy, filename = "VIOutputStationary.gif") 
#         else :
#             env.get_gif(policy, filename = "ModifiedVIOutputStationary.gif") 
#     elif degrade_pitch == True:
        
#         if passTimeStamp == True:
#             env.get_gif(policy, filename = "VIOutputNonStationary.gif")
#         else:
#             env.get_gif(policy, filename = "VIOutputNonStationaryWithoutPassingTimeStamp.gif")
    
    print("Count of total number of calls made to the  env.get_transitions_at_time is : ", numCallsToGetTransition)
    
    return policy, valueFn, 1, numIterations, numCallsToGetTransition