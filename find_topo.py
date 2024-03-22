# this is a test of reconstructing topology of network
# based on the idea of q-learning
# In this edition, we change q-table from a DataFrame to a dict
# instead of storing all the possible states, we only store when calculated
# which means no update in further steps (this should greatly reduce space complexity)

# besides, for time complexity, we redesign 'episode'
# in the initial plan, for each episode, we keep finding possible states
# until a situation that (all other states are already found)
# after that, an episode is terminated

# to solve the problem of local-maximum, we will randomly find a state that not yet dicovered
# keep going, meanwhile tracking the state with the highest score

# notice that action is no longer stored, it will directly calculated by choosing states
# only legal states are further considered

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# creating q table
import pandas as pd
from itertools import combinations

# quantify RS and overload
from key_functions.quantify_topo import *

import random

import time

# node coordinations
# simple demonstration
UAV_coords = np.array([
    # (250,200,200),
    # (250,600,200),
    # (600,350,200),

    # (588, 127, 246),
    # (665, 310, 180),
    # (428, 777, 201),
    (513, 769, 193),
    (548, 317, 216),

    (783, 626, 235),
    (411, 254, 224),
    (600, 725, 224),
    # (419, 38, 151),
    # (423, 215, 183),
    # (643, 641, 198),
])

ABS_coords = np.array([
    # (440,390,500),

    # (294, 467, 500),
    # (445, 0, 500),

    (511, 133, 500),
    (244, 637, 500),
])

reward_hyper = {
    'DRPenalty': 0.5,
    'BPHopConstraint': 4,
    'BPDRConstraint': 100000000,
    'droppedRatio': 0.2,
    'ratioDR': 0.6,
    'ratioBP': 0.4,
    'weightDR': 0.3,
    'weightBP': 0.4,
    'weightNP': 0.3,
    'overloadConstraint': 10000
}

# Get reward of a state, including resilience score and optimization score
def Reward(state):
    # notice that score = RS-overload
    # or RS*overload

    UAVMap = get_UAVMap(state=state, UAV_position=UAV_coords, ABS_position=ABS_coords)

    # Unpack hyperparameters from the dictionary
    DRPenalty = reward_hyper['DRPenalty']
    BPHopConstraint = reward_hyper['BPHopConstraint']
    BPDRConstraint = reward_hyper['BPDRConstraint']
    droppedRatio = reward_hyper['droppedRatio']
    ratioDR = reward_hyper['ratioDR']
    ratioBP = reward_hyper['ratioBP']
    weightDR = reward_hyper['weightDR']
    weightBP = reward_hyper['weightBP']
    weightNP = reward_hyper['weightNP']
    overloadConstraint = reward_hyper['overloadConstraint']

    ResilienceScore = get_RS(UAVMap, DRPenalty, BPHopConstraint, BPDRConstraint, droppedRatio, ratioDR, ratioBP, weightDR, weightBP, weightNP)

    # as for the reward function, we need also to consider the balance in the UAV network
    # here we use gini coefficient
    overloadConstraint = 10000
    OverloadScore = measure_overload(UAVMap, BPHopConstraint, BPDRConstraint, overloadConstraint)

    # now we just return RS*overload
    rewardScore = ResilienceScore*OverloadScore

    return rewardScore, ResilienceScore, OverloadScore
    # return rewardScore

def generate_adjacent_states(state):
    adjacent_states = []

    for i in range(len(state)):
        # Convert the string into a list so that we can modify it
        state_list = list(state)

        # Change the current bit 
        # (if it's '0' change it to '1', if it's '1' change it to '0')
        state_list[i] = '1' if state[i] == '0' else '0'

        # Convert the modified list back into a string and add it to the result list
        new_state = ''.join(state_list)
        adjacent_states.append(new_state)

    return adjacent_states

def process_states(adjacent_states, q_table):
    next_state_sum = len(adjacent_states)
    next_state_all = {}
    
    for state in adjacent_states:
        if state in q_table:
            next_state_all[state] = q_table[state]
            next_state_sum -= 1
        else:
            next_state_score = Reward(state)
            next_state_all[state] = next_state_score
            q_table[state] = next_state_score
    return next_state_all, next_state_sum > 0
    
def take_action(state_scores, epsilon):
    # Check if state_scores is empty
    if not state_scores:
        return None

    # Get all key-value pairs where the first value is non-zero
    non_zero_items = {k: v for k, v in state_scores.items() if v[0] != 0}

    # If all first values are zero, return None
    if not non_zero_items:
        return None

    if random.random() < epsilon:
        # With probability epsilon, randomly select a key-value pair where the first value is non-zero
        return random.choice(list(non_zero_items.items()))
    else:
        # With probability 1-epsilon, select the key-value pair with the largest first value
        max_key = max(non_zero_items, key=lambda k: non_zero_items[k][0])
        return max_key, non_zero_items[max_key]

def generate_random_binary_string(input_string):
    length = len(input_string)
    random_string = ''.join(random.choice(['0', '1']) for _ in range(length))
    return random_string

# Q-learning hyperparameters
epsilon = 0.1
# randomness of choosing actions
best_state = ""

q_table = {}

reward_track = []
RS_track = []
OL_track = []

max_reward = 0
num_nodes = len(ABS_coords) + len(UAV_coords)

state = '0' * int((num_nodes*(num_nodes-1)/2))
# state = '1' * int((num_nodes*(num_nodes-1)/2))

start_time = time.time()

for episode in range(50):
    next_possible_states = generate_adjacent_states(state)
    states_scores, end_flag = process_states(next_possible_states, q_table)

    next_state, next_state_score = take_action(states_scores, epsilon)

    if next_state_score[0] > max_reward:
        max_reward = next_state_score[0]
        best_state = next_state
        # print("Q")
    
    # print(episode)
    # reward_track.append(max_reward)
    reward_track.append(next_state_score[0])
    RS_track.append(next_state_score[1])
    OL_track.append(next_state_score[2])

    if not end_flag: 
        state = generate_random_binary_string(state)
        # break
        continue

end_time = time.time()
elapsed_time = end_time - start_time
print(f"The code block ran in {elapsed_time} seconds")


# print(generate_adjacent_states('00110011011100'))
# print(generate_adjacent_states('01001110'))
# q_table = {'01101': 0.745, '10011': 0.658}
# input_state = '01111'
# adjacent_states = generate_adjacent_states(input_state)
# print(process_states(adjacent_states, q_table))

        
print(best_state)
print(max_reward)

# visualization
plt.plot(reward_track, label='Reward', color='blue')
plt.plot(RS_track, label='RS Value', color='red')
plt.plot(OL_track, label='OL Value', color='green') 

plt.title('Track Values Over Episodes')  # set title
plt.xlabel('Episode')  # set x label
plt.ylabel('Value')  # set y label
plt.legend()  # show legend to distinguish tracks

plt.show()