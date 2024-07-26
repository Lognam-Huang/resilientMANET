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
# UAV_coords = np.array([
#     # (250,200,200),
#     # (250,600,200),
#     # (600,350,200),

#     # (588, 127, 246),
#     # (665, 310, 180),
#     # (428, 777, 201),
#     (513, 769, 193),
#     (548, 317, 216),

#     (783, 626, 235),
#     (411, 254, 224),
#     # (600, 725, 224),
#     # (419, 38, 151),
#     # (423, 215, 183),
#     # (643, 641, 198),
# ])

# ABS_coords = np.array([
#     # (440,390,500),

#     # (294, 467, 500),
#     # (445, 0, 500),

#     (511, 133, 500),
#     (244, 637, 500),
# ])

# reward_hyper = {
#     'DRPenalty': 0.5,
#     'BPHopConstraint': 4,
#     'BPDRConstraint': 100000000,
#     'droppedRatio': 0.2,
#     'ratioDR': 0.6,
#     'ratioBP': 0.4,
#     'weightDR': 0.3,
#     'weightBP': 0.4,
#     'weightNP': 0.3,
#     'overloadConstraint': 10000
# }

from key_functions.quantify_topo import quantify_data_rate_with_GU, quantify_backup_path_with_GU, quantify_network_partitioning_with_GU
from simu_functions import calculate_capacity_and_overload, get_gu_to_uav_connections
from classes.UAVMap import find_best_paths_to_bs
from functions.print_nodes import get_nodes_position, print_nodes

# Get reward of a state, including resilience score and optimization score
def Reward(state, scene_info, GU_nodes, UAV_nodes, ABS_coords, reward_hyper):
    # notice that score = RS-overload
    # or RS*overload

    # print_nodes(UAV_nodes)
    UAV_coords = np.array(get_nodes_position(UAV_nodes))
    

    UAVMap = get_UAVMap(state=state, UAV_position= UAV_coords, ABS_position=ABS_coords, scene_info=scene_info)

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

    uav_to_bs_connections = find_best_paths_to_bs(UAVMap)
    gu_to_uav_connections = get_gu_to_uav_connections(GU_nodes, UAV_nodes, scene_info['UAV'], scene_info['blocks'])

    gu_to_bs_capacity, uav_to_bs_capacity, uav_overload = calculate_capacity_and_overload(
        GU_nodes, gu_to_uav_connections, uav_to_bs_connections, scene_info['UAV'], UAVMap, UAV_nodes
    )

    # 计算 RS（Resilience Score）
    ResilienceScore = get_RS_with_GU(
        GU_nodes, gu_to_uav_connections, UAVMap, 
        reward_hyper['DRPenalty'], reward_hyper['BPHopConstraint'], reward_hyper['BPDRConstraint'], 
        reward_hyper['droppedRatio'], reward_hyper['ratioDR'], reward_hyper['ratioBP'], 
        reward_hyper['weightDR'], reward_hyper['weightBP'], reward_hyper['weightNP'], 
        scene_info, gu_to_bs_capacity
    )

    # 计算 OL（Overload Score）
    OverloadScore = measure_overload_with_GU(uav_overload)

    # 计算 reward 分数
    # rewardScore = ResilienceScore * OverloadScore
    rewardScore = ResilienceScore 
    # ResilienceScore = get_RS(UAVMap, DRPenalty, BPHopConstraint, BPDRConstraint, droppedRatio, ratioDR, ratioBP, weightDR, weightBP, weightNP, scene_info)

    # # as for the reward function, we need also to consider the balance in the UAV network
    # # here we use gini coefficient
    # overloadConstraint = 10000
    # OverloadScore = measure_overload(UAVMap, BPHopConstraint, BPDRConstraint, overloadConstraint, scene_info)

    # # now we just return RS*overload
    # rewardScore = ResilienceScore*OverloadScore

    # Lognam: try to make sure every UAV has a path towards BS, directly or indirectly
    if not all_uavs_connected_to_abs(UAVMap, len(UAV_coords)):
        rewardScore *= 0.5

    return rewardScore, ResilienceScore, OverloadScore, UAVMap
    # return rewardScore

def all_uavs_connected_to_abs(UAVMap, num_uavs):
    def bfs(start, targets, graph):
        visited = set()
        queue = [start]
        
        while queue:
            node = queue.pop(0)
            if node in targets:
                return True
            if node not in visited:
                visited.add(node)
                queue.extend(graph.get(node, []))
        
        return False

    # 初始化UAV和BS的集合
    uav_set = set(range(num_uavs))
    bs_set = set()

    # 初始化图，包含所有UAV和可能的BS节点
    graph = {i: [] for i in range(num_uavs)}

    # 提取所有路径中的BS节点
    for uav, paths in UAVMap.allPaths.items():
        for path_info in paths:
            path = path_info['path']
            for node in path:
                if node >= num_uavs:
                    bs_set.add(node)
    
    # 确保图中包含所有BS节点
    for bs in bs_set:
        graph[bs] = []

    # 从UAVMap中提取路径信息并构建图
    for uav, paths in UAVMap.allPaths.items():
        for path_info in paths:
            path = path_info['path']
            for i in range(len(path) - 1):
                graph[path[i]].append(path[i + 1])
                graph[path[i + 1]].append(path[i])

    # 检查每个UAV是否能连接到任意一个BS
    for uav in uav_set:
        if not bfs(uav, bs_set, graph):
            return False

    return True


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

def process_states(adjacent_states, q_table, scene_info, GU_nodes, UAV_nodes, ABS_coords, reward_hyper):
    next_state_sum = len(adjacent_states)
    next_state_all = {}
    
    for state in adjacent_states:
        if state in q_table:
            next_state_all[state] = q_table[state]
            next_state_sum -= 1
        else:
            next_state_score = Reward(state, scene_info, GU_nodes, UAV_nodes, ABS_coords, reward_hyper)
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

def find_best_topology(GU_nodes, UAV_nodes, ABS_coords, eps, reward_hyper, episodes=50, visualize=False, scene_info = None, print_prog = False):
    best_state = ""
    q_table = {}
    reward_track = []
    RS_track = []
    OL_track = []
    max_reward = 0
    best_RS = 0
    best_OL = 0

    # UAV_coords = np.array(get_nodes_position(UAV_nodes))
    
    num_nodes = len(ABS_coords) + len(UAV_nodes)
    state = '0' * int((num_nodes * (num_nodes - 1) / 2))
    start_time = time.time()

    best_state_UAVMap = None

    epsilon = eps

    for episode in range(episodes):
        next_possible_states = generate_adjacent_states(state)
        states_scores, end_flag = process_states(next_possible_states, q_table, scene_info, GU_nodes, UAV_nodes, ABS_coords, reward_hyper)

        next_state, next_state_score = take_action(states_scores, epsilon)

        if next_state_score[0] > max_reward:
            best_state = next_state
            max_reward = next_state_score[0]
            best_RS = next_state_score[1]
            best_OL = next_state_score[2]
            best_state_UAVMap = next_state_score[3]

        reward_track.append(next_state_score[0])
        RS_track.append(next_state_score[1])
        OL_track.append(next_state_score[2])

        if not end_flag:
            state = generate_random_binary_string(state)
            continue

    if print_prog:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The code block ran in {elapsed_time} seconds")
        print(f"Best state: {best_state}")
        print(f"Max reward: {max_reward}")

    if visualize:
        # visualization
        plt.plot(reward_track, label='Reward', color='blue')
        plt.plot(RS_track, label='RS Value', color='red')
        plt.plot(OL_track, label='OL Value', color='green')
        plt.title('Track Values Over Episodes')  # set title
        plt.xlabel('Episode')  # set x label
        plt.ylabel('Value')  # set y label
        plt.legend()  # show legend to distinguish tracks
        plt.show()

    return best_state, max_reward, best_RS, best_OL, reward_track, RS_track, OL_track, best_state_UAVMap



if __name__ == "__main__":
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