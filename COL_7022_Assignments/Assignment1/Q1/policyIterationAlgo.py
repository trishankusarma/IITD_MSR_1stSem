from env import FootballSkillsEnv
from model import model
import numpy as np

def policyIterationAlgo(envr=FootballSkillsEnv, model=model, logEnabled = True, degrade_pitch = False, discount_factor = 0.95):
    print("Starting Policy Iteration Algorithm")
    '''
    Implements the Policy Iteration algorithm to find the optimal policy for the 
    Football Skills Environment.
    
    Degrade_pitch -> False -> stationary -> policy only depends on state
    Degrade_pitch -> True -> non-stationary -> policy depends on both state and time
    
    Args:
        envr (class, optional): Environment class to instantiate. Defaults to FootballSkillsEnv.
    
    Returns:
        tuple: (optimal_policy, value_function, num_iterations)
            - optimal_policy (numpy.ndarray): each index maps a state to optimal actions
            - value_function (numpy.ndarray): Value of each state under optimal policy  
            - num_iterations (int): Number of iterations until convergence
    
    Algorithm:
    1. Initialize arbitrary policy and value function
    2. Policy Evaluation: Iteratively update value function until convergence
    3. Policy Improvement: Update policy greedily based on current values  
    4. Repeat steps 2-3 until policy converges
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
    numOuterIterations = 0
    numInnerIterations = 0
    
    # initialize policy
    policy, valueFn = model.initialize(degrade_pitch)
    # policy will say for any state, which is the right action to perform
    # valueFn will be give the sum of the accumulated rewards from step s 

    # EM algo for policy improvement
    while(True):

        numOuterIterations += 1
        
        # Policy Evaluation
        valueFn, numInnerIterationsCurr = model.performPolicyEvaluationForPI(allStateTuples, policy, valueFn, degrade_pitch, env)
        numInnerIterations += numInnerIterationsCurr

        # Policy Improvement
        policy, policyStable = model.performPolicyImprovementForPI(allStateTuples, policy, valueFn, actionIndexes, degrade_pitch, env)
        if policyStable == True:
            break
    
    # 6
    if degrade_pitch == False:
        env.get_gif(policy, filename = "PIOutputStationary.gif") 
        callsToGetTransistion = len(allStateTuples)*numInnerIterations + len(allStateTuples)*len(actionIndexes)*numOuterIterations
    else:
        env.get_gif(policy, filename = "PIOutputNonStationary.gif") 
        callsToGetTransistion = len(allStateTuples)*(model.non_stationary_horizon)*numInnerIterations + len(allStateTuples)*(model.non_stationary_horizon)*len(actionIndexes)*numOuterIterations
        
    print("Count of total number of calls made to the  env.get_transitions_at_time is : ", callsToGetTransistion)
        
    return policy, valueFn, numOuterIterations, numInnerIterations, callsToGetTransistion