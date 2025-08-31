from env import FootballSkillsEnv
from model import model
import numpy as np

def policyIterationAlgo(envr=FootballSkillsEnv, model=model, logEnabled = True, degrade_pitch = False):
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
    env = envr(render_mode='gif')
    
    # get all states
    allStateTuples = model.generateAllStates(env)
    actionIndexes = model.generateAllActionIndexes(env)
    numOuterIterations = 0
    numInnerIterations = 0
    
    # initialize policy
    policy, valueFn = model.initialize()
    # policy will say for any state, which is the right action to perform
    # valueFn will be give the sum of the accumulated rewards from step s 

    # EM algo for policy improvement
    while(True):

        numOuterIterations += 1
        
        # Policy Evaluation
        valueFn, numInnerIterationsCurr = model.performPolicyEvaluation(allStateTuples, policy, valueFn, env)
        numInnerIterations += numInnerIterationsCurr

        # Policy Improvement
        policy, policyStable = model.performPolicyImprovement(allStateTuples, policy, valueFn, actionIndexes, env)
        if policyStable == True:
            break
    
    callsToGetTransistion = len(allStateTuples)*numInnerIterations + len(allStateTuples)*len(actionIndexes)*numOuterIterations
    print("Count of total number of calls made to the  env.get_transitions_at_time is : ", callsToGetTransistion)
    
    # 6
    env.get_gif(policy, filename = "PIOutput.gif") 
    return policy, valueFn, numOuterIterations, numInnerIterations, callsToGetTransistion