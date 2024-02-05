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
from quantify_topo import *

import random

# node coordinations
# simple demonstration
UAV_coords = np.array([
    # (250,200,200),
    # (250,600,200),
    # (600,350,200),

    (588, 127, 246),
    (665, 310, 180),
    (428, 777, 201),
    (513, 769, 193),
    (548, 317, 216),

    (783, 626, 235),
    (411, 254, 224),
    (600, 725, 224),
    (419, 38, 151),
    (423, 215, 183),
    (643, 641, 198),
])

ABS_coords = np.array([
    # (440,390,500),

    (294, 467, 500),
    (445, 0, 500),

    (511, 133, 500),
    (244, 637, 500),
])

q_table = {}

# Get reward of a state, including resilience score and optimization score
def Reward(state):
    # notice that score = RS-overload
    # or RS*overload

    UAVMap = get_UAVMap(state=state, UAV_position=UAV_coords, ABS_position=ABS_coords)

    # quantify resilience score: data rate
    DRPenalty = 0.5

    # quantify resilience score: backup path 
    BPHopConstraint = 4
    BPDRConstraint = 100000000

    # quantify resilience score: network partitioning
    droppedRatio = 0.2
    ratioDR = 0.6
    ratioBP = 0.4

    # integrate quantificaiton
    weightDR = 0.3
    weightBP = 0.4
    weightNP = 0.3

    ResilienceScore = get_RS(UAVMap, DRPenalty, BPHopConstraint, BPDRConstraint, droppedRatio, ratioDR, ratioBP, weightDR, weightBP, weightNP)
    # print("Resilience score is:")
    # print(ResilienceScore) 

    # as for the reward function, we need also to consider the balance in the UAV network
    # here we use gini coefficient
    overloadConstraint = 10000
    OverloadScore = measure_overload(UAVMap, BPHopConstraint, BPDRConstraint, overloadConstraint)
    # print("Overload score is:")
    # print(OverloadScore)

    # now we just return RS*overload
    rewardScore = ResilienceScore*OverloadScore

    # print("Reward score is:")
    # print(rewardScore)

    return rewardScore

def generate_adjacent_states(state):
    adjacent_states = []

    for i in range(len(state)):
        # 将字符串转换为列表，以便我们可以修改它
        state_list = list(state)

        # 改变当前位（如果是 '0' 改为 '1'，如果是 '1' 改为 '0'）
        state_list[i] = '1' if state[i] == '0' else '0'

        # 将修改后的列表转换回字符串，并添加到结果列表中
        new_state = ''.join(state_list)
        adjacent_states.append(new_state)

        # print(''.join(state_list))
        # print(Reward(new_state))

    return adjacent_states

def process_states(adjacent_states, q_table):

    # print(len(adjacent_states))
    next_state_sum = len(adjacent_states)
    next_state_all = {}
    
    for state in adjacent_states:
        if state in q_table:
            # print(f"State: {state}, Score: {q_table[state]}")
            next_state_all[state] = q_table[state]
            next_state_sum -= 1
        else:
            # q_table[state] = Reward(state)

            next_state_score = Reward(state)
            next_state_all[state] = next_state_score
            q_table[state] = next_state_score
            # print(f"Added new state {state} with score 1")
    return next_state_all, next_state_sum > 0

def take_action(a, b):
    # 检查 a 是否为空
    if not a:
        return None

    # 获取所有值非零的键值对
    non_zero_items = {k: v for k, v in a.items() if v != 0}

    # 如果所有项的值都为零，返回 None
    if not non_zero_items:
        return None

    if random.random() < b:
        # 在 b 的概率下，随机选择一个值非零的键值对
        return random.choice(list(non_zero_items.items()))
    else:
        # 在 1-b 的概率下，选择具有最大值的键值对
        max_key = max(non_zero_items, key=non_zero_items.get)
        return max_key, non_zero_items[max_key]

def generate_random_binary_string(input_string):
    length = len(input_string)
    random_string = ''.join(random.choice(['0', '1']) for _ in range(length))
    return random_string

# Q-learning hyperparameters
epsilon = 0.1
# randomness of choosing actions
best_state = ""
reward_track = []

max_reward = 0
num_nodes = len(ABS_coords) + len(UAV_coords)

state = '0' * int((num_nodes*(num_nodes-1)/2))
# state = '1' * int((num_nodes*(num_nodes-1)/2))

for episode in range(5):
    # print(episode)
    # print(state)
    while True:
        next_possible_states = generate_adjacent_states(state)
        states_score, end_flag = process_states(next_possible_states, q_table)

        if not end_flag: 
            # next_state = generate_random_binary_string(state)
            state = generate_random_binary_string(state)
            break
            # continue

        next_state, next_state_score = take_action(states_score, epsilon)

        # print(next_state)
        # print(next_state_score)

        if next_state_score > max_reward:
            max_reward = next_state_score
            best_state = next_state
            print("Q")
        
        reward_track.append(max_reward)


# print(generate_adjacent_states('00110011011100'))
# print(generate_adjacent_states('01001110'))
# q_table = {'01101': 0.745, '10011': 0.658}
# input_state = '01101'
# adjacent_states = generate_adjacent_states(input_state)
# print(process_states(adjacent_states, q_table))

# # 查看更新后的 q_table
# print("\nUpdated q_table:")
# print(q_table)

# num_nodes = len(node_coords)
        
# print(best_state)
# print(max_reward)

# print(reward_track)
# print(q_table)

# 可视化RS值
plt.plot(reward_track)
plt.title('RS Value Over Episodes')
plt.xlabel('Episode')
plt.ylabel('RS Value')
plt.show()
