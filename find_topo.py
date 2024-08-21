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

from key_functions.quantify_topo import quantify_data_rate_with_GU, quantify_backup_path_with_GU, quantify_network_partitioning_with_GU
from simu_functions import calculate_capacity_and_overload, get_gu_to_uav_connections
from classes.UAVMap import find_best_paths_to_bs
from functions.print_nodes import get_nodes_position, print_nodes

# Get reward of a state, including resilience score and optimization score
def Reward(state, scene_info, GU_nodes, UAV_nodes, ABS_coords, reward_hyper):

    UAV_coords = np.array(get_nodes_position(UAV_nodes))
    
    UAVMap = get_UAVMap(state=state, UAV_position= UAV_coords, ABS_position=ABS_coords, scene_info=scene_info)

    uav_to_bs_connections = find_best_paths_to_bs(UAVMap)
    gu_to_uav_connections = get_gu_to_uav_connections(GU_nodes, UAV_nodes, scene_info['UAV'], scene_info['blocks'])

    gu_to_bs_capacity, uav_to_bs_capacity, uav_overload = calculate_capacity_and_overload(
        GU_nodes, gu_to_uav_connections, uav_to_bs_connections, scene_info['UAV'], UAVMap, UAV_nodes, scene_info['blocks']
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
    
    min_RS_with_one_bs_removed = ResilienceScore  

    # If there are multiple BS, proceed to test each one being "nullified"
    if len(ABS_coords) > 1:
        for i in range(len(ABS_coords)):
            # 将与被移除的BS相关的边设为0
            modified_state = disable_bs_edges_in_state(state, i, len(UAV_nodes), len(ABS_coords))

            # 重新计算UAVMap，使用修改后的状态
            modified_UAVMap = get_UAVMap(state=modified_state, UAV_position=UAV_coords, ABS_position=ABS_coords, scene_info=scene_info)

            # 重新计算uav_to_bs_connections
            modified_uav_to_bs_connections = find_best_paths_to_bs(modified_UAVMap)

            # 重新计算GU-to-BS capacity和UAV overload
            modified_gu_to_bs_capacity, modified_uav_to_bs_capacity, modified_uav_overload = calculate_capacity_and_overload(
                GU_nodes, gu_to_uav_connections, modified_uav_to_bs_connections, scene_info['UAV'], modified_UAVMap, UAV_nodes, scene_info['blocks']
            )

            # 重新计算ResilienceScore
            RS_with_one_bs_removed = get_RS_with_GU(
                GU_nodes, gu_to_uav_connections, modified_UAVMap,
                reward_hyper['DRPenalty'], reward_hyper['BPHopConstraint'], reward_hyper['BPDRConstraint'],
                reward_hyper['droppedRatio'], reward_hyper['ratioDR'], reward_hyper['ratioBP'],
                reward_hyper['weightDR'], reward_hyper['weightBP'], reward_hyper['weightNP'],
                scene_info, modified_gu_to_bs_capacity
            )

            # 记录最小的ResilienceScore
            min_RS_with_one_bs_removed = min(min_RS_with_one_bs_removed, RS_with_one_bs_removed)

        # 计算鲁棒性因子
        robustness_factor = (min_RS_with_one_bs_removed / ResilienceScore if ResilienceScore > 0 else 0)
        rewardScore *= robustness_factor  # Adjust the original RS

        


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
 
import random

def take_action(state_scores, epsilon):
    """
    在考虑epsilon的情况下选择下一个状态。
    - 以epsilon的概率随机选择一个状态。
    - 以(1-epsilon)的概率选择具有最大reward的状态。
    如果所有reward都为0，则直接随机选择一个状态。

    Parameters:
    state_scores: 字典，键为状态，值为与该状态相关的分数元组（如 (reward, ...)）。
    epsilon: 探索的概率。

    Returns:
    next_state: 被选择的下一个状态。
    next_state_score: 被选择状态对应的分数。
    """
    # 如果state_scores为空，直接返回None
    if not state_scores:
        return None, None

    # 获取所有非零reward的状态
    non_zero_items = {k: v for k, v in state_scores.items() if v[0] != 0}

    if non_zero_items:
        # 如果非零reward的状态存在
        if random.random() < epsilon:
            # 以epsilon的概率随机选择一个状态
            next_state = random.choice(list(state_scores.keys()))
        else:
            # 以(1-epsilon)的概率选择具有最大reward的状态
            max_key = max(non_zero_items, key=lambda k: non_zero_items[k][0])
            next_state = max_key
    else:
        # 如果所有reward都为0，随机选择一个状态
        next_state = random.choice(list(state_scores.keys()))

    next_state_score = state_scores[next_state]
    return next_state, next_state_score


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
    # state = '0' * int((num_nodes * (num_nodes - 1) / 2))
    state = '1' * int((num_nodes * (num_nodes - 1) / 2))
    
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

def disable_bs_edges_in_state(state, bs_index, num_uavs, num_bss):
    """
    将与被“无效”BS相关的边在状态字符串中设置为0。
    
    Parameters:
    state: 原始的状态字符串。
    bs_index: 要被“无效”的BS的索引（0-based）。
    num_uavs: UAV的数量。
    num_bss: BS的总数。

    Returns:
    modified_state: 状态字符串，其中与指定BS相关的边被设置为0。
    """
    num_nodes = num_uavs + num_bss
    bs_start_index = num_uavs + bs_index  # 与BS相关的节点索引
    
    modified_state = list(state)  # 将状态字符串转换为列表以便修改
    
    # 将与该BS相关的所有边设置为0
    for i in range(bs_start_index):
        edge_index = edge_index_in_state(i, bs_start_index, num_nodes)
        modified_state[edge_index] = '0'
    
    return ''.join(modified_state)

def edge_index_in_state(node1, node2, num_nodes):
    """
    根据两个节点的索引，返回状态字符串中对应的边的索引。
    
    Parameters:
    node1: 第一个节点的索引（0-based）。
    node2: 第二个节点的索引（0-based）。
    num_nodes: 节点的总数。

    Returns:
    edge_index: 状态字符串中对应边的索引。
    """
    if node1 > node2:
        node1, node2 = node2, node1
    return int(node1 * (2 * num_nodes - node1 - 1) / 2 + (node2 - node1 - 1))

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