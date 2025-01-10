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

def process_states(adjacent_states, q_table, scene_info, GU_nodes, UAV_nodes, ABS_nodes, reward_hyper):
    next_state_sum = len(adjacent_states)
    next_state_all = {}

    for state in adjacent_states:
        if state in q_table:
            next_state_all[state] = q_table[state]
            next_state_sum -= 1
        else:
            next_state_score = reward(state, scene_info, GU_nodes, UAV_nodes, ABS_nodes, reward_hyper)
            next_state_all[state] = next_state_score
            q_table[state] = next_state_score
    return next_state_all, next_state_sum > 0

import random

def generate_random_binary_string(input_string):
    length = len(input_string)
    random_string = ''.join(random.choice(['0', '1']) for _ in range(length))
    return random_string

def take_action(state_scores, epsilon):
    if not state_scores:
        return None, None

    non_zero_items = {k: v for k, v in state_scores.items() if v[0] != 0}

    if non_zero_items:
        if random.random() < epsilon:
            next_state = random.choice(list(state_scores.keys()))
        else:
            max_key = max(non_zero_items, key=lambda k: non_zero_items[k][0])
            next_state = max_key
    else:
        next_state = random.choice(list(state_scores.keys()))

    next_state_score = state_scores[next_state]
    return next_state, next_state_score

# Get reward of a state, including resilience score and optimization score
def reward(state, scene_info, GU_nodes, UAV_nodes, BS_nodes, reward_hyper):
    print("Current target state: "+str(state))

    time_counter = time.time()

    backhaul_connection = get_backhaul_connection(state=state, UAV_nodes= UAV_nodes, BS_nodes=BS_nodes, scene_info=scene_info)

    print(f"It takes {time.time()-time_counter} seconds to calculate backhaul connection.")
    time_counter = time.time()

    resilience_score = get_RS(GU_nodes, UAV_nodes, backhaul_connection, reward_hyper, scene_info)
    print("Original RS is: "+str(resilience_score))

    print(f"It takes {time.time()-time_counter} seconds to calculate RS.")
    time_counter = time.time()

    reward_score = resilience_score

    # Lognam: try to make sure every UAV has a path towards BS, directly or indirectly
    for start_point, paths in backhaul_connection.allPaths.items():
        if not paths:
            reward_score *= 0.5

    print(f"It takes {time.time()-time_counter} seconds to calculate constraint 1.")
    time_counter = time.time()

    min_reward_score_with_one_bs_removed = reward_score

    # If there are multiple BS, proceed to test each one being "nullified"
    if len(BS_nodes) > 1:
        for i in range(len(BS_nodes)):
            modified_state = disable_bs_edges_in_state(state, i, len(UAV_nodes))

            modified_backhaul_connection = get_backhaul_connection(state=modified_state, UAV_nodes= UAV_nodes, BS_nodes=BS_nodes, scene_info=scene_info)

            modified_resilience_score = get_RS(GU_nodes, UAV_nodes, modified_backhaul_connection, reward_hyper, scene_info)

            min_reward_score_with_one_bs_removed = min(min_reward_score_with_one_bs_removed,  modified_resilience_score)

        robustness_factor = (min_reward_score_with_one_bs_removed / resilience_score if resilience_score > 0 else 0)
        reward_score *= robustness_factor  # Adjust the original RS

    print(f"It takes {time.time()-time_counter} seconds to calculate constraint 2.")
    time_counter = time.time()

    return reward_score, resilience_score, backhaul_connection

from classes.BackhaulPaths import BackhaulPaths
def get_backhaul_connection(state, UAV_nodes, BS_nodes, scene_info = None):
    blocks = scene_info['blocks']
    UAVInfo = scene_info['UAV']
    scene = scene_info['scenario']
        
    # we should set connection based on the state 
    set_connected_edges(''.join(str(int(i)) for i in state), UAV_nodes, BS_nodes)

    backhaul_connection = BackhaulPaths(UAV_nodes, BS_nodes, blocks, UAVInfo)

    return backhaul_connection

from itertools import combinations

def set_connected_edges(state, UAV_nodes, BS_nodes):

    a = len(UAV_nodes)
    b = len(BS_nodes)

    L = len(state)  
    n = (1 + math.sqrt(1 + 8 * L)) / 2
    
    if n != a+b or L != ((a+b)*(a+b-1)/2):
        ValueError("Invalid number of nodes or state")
    
    n = int(n)  
    edges = []  
    node_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]

    for index, pair in enumerate(node_pairs):
        if state[index] == '1':
            edges.append(pair)
    
    for uav in UAV_nodes:
        uav.reset_connection()
    for bs in BS_nodes:
        bs.reset_connection()

    for start_node, end_node in edges:
        if end_node < a:
            UAV_nodes[start_node].add_connection(end_node)
            UAV_nodes[end_node].add_connection(start_node)
        elif end_node < a+b:
            if start_node < a:
                BS_nodes[end_node-a].add_connection(start_node)
            else:
                BS_nodes[end_node-a].add_bs_connection(start_node)
            # BS_nodes[end_node].add_connection(start_node)
        else:
            ValueError
    
    return edges


from node_functions import print_node
def get_RS(GU_nodes, UAV_nodes, backhaul_connection, reward_hyper, scene_info):
    UAVInfo = scene_info['UAV']

    DRScore = quantify_data_rate(GU_nodes, backhaul_connection, reward_hyper['DRPenalty'], UAVInfo)
    BPScore = quantify_backup_path(GU_nodes, UAV_nodes, backhaul_connection, reward_hyper['BPHopConstraint'], reward_hyper['BPDRConstraint'], scene_info)
    NPScore = quantify_network_partitioning(GU_nodes, UAV_nodes, backhaul_connection, reward_hyper['droppedRatio'], reward_hyper['DRPenalty'], reward_hyper['BPHopConstraint'], reward_hyper['BPDRConstraint'], UAVInfo, DRScore, BPScore, reward_hyper['ratioDR'], reward_hyper['ratioBP'], scene_info)

    print(DRScore)
    print(BPScore)
    print(NPScore)

    ResilienceScore = integrate_quantification(DRScore, BPScore, NPScore, reward_hyper['weightDR'], reward_hyper['weightBP'], reward_hyper['weightNP'])
    return ResilienceScore

def quantify_data_rate(ground_users, backhaul_connection, r, UAVInfo):
    data_rates = list()

    for gu in ground_users:
        gu_to_BS_bottleneck = min(float(gu.data_rate[0]), max((path['DR'] for path in backhaul_connection.allPaths[gu.connected_nodes[0]]), default=0))
        data_rates.append(gu_to_BS_bottleneck)

    min_DR = min(data_rates)
    avg_DR = sum(data_rates) / len(data_rates)

    def norm(score, UAVInfo):
        normScore = min(score / UAVInfo['bandwidth'], 1)
        return normScore

    score = r * norm(min_DR, UAVInfo) + (1 - r) * norm(avg_DR, UAVInfo)
    return score

from functions.path_is_blocked import path_is_blocked
from functions.calculate_data_rate import calculate_data_rate

def quantify_backup_path(ground_users, UAV_nodes, backhaul_connection, hop_constraint, DR_constraint, scene_info):
    UAVInfo = scene_info['UAV']
    blocks = scene_info['blocks']
    best_paths = {}

    for gu in ground_users:
        filtered_paths = [p for p in backhaul_connection.allPaths[gu.connected_nodes[0]] if len(p['path']) <= hop_constraint and  min(float(gu.data_rate[0]), p['DR']) >= DR_constraint]
        if filtered_paths:
            best_path = max(filtered_paths, key=lambda p: p['DR'])
            best_paths[gu.node_number] = (best_path['DR'], len(best_path['path']))
        else:
            best_paths[gu.node_number] = (0, float('inf'))
    total_score = 0

    # print(best_paths)
    # print("QBK is called once")

    for gu in ground_users:
        for uav in UAV_nodes:
            gu_and_uav_is_blocked = path_is_blocked(blocks, uav, gu)
            if not gu_and_uav_is_blocked:
                dr_from_gu_to_uav = calculate_data_rate(UAVInfo, uav.position, gu.position,  gu_and_uav_is_blocked)
            else:
                continue
            best_path_DR, best_path_hop = best_paths[gu.node_number]
            paths = backhaul_connection.allPaths[uav.node_number]
            for path in paths:
                gu_to_bs_bottleneck = min(dr_from_gu_to_uav, path['DR'])
                if len(path['path']) <= hop_constraint and gu_to_bs_bottleneck >= DR_constraint:
                    # if gu_to_bs_bottleneck >= best_path_DR or best_path == 0:
                    if gu_to_bs_bottleneck >= best_path_DR:
                        total_score += 1
                    else:
                        hop_difference = len(path['path']) - best_path_hop
                        if hop_difference <= 0:
                            total_score += gu_to_bs_bottleneck /best_path_DR
                        else:
                            total_score += (gu_to_bs_bottleneck / best_path_DR) / hop_difference
                else:
                    # print("Should be called frequently")
                    # total_score -= 1
                    # following part is changed because of hard scene: there are maybe too many paths, whose DR is acceptable meanwile hop too much
                    if gu_to_bs_bottleneck < DR_constraint:
                        # print("Hope not to be called frequently")
                        total_score -= 1
                

        # for p in backhaul_connection.allPaths[gu.connected_nodes[0]]:
        #     total_path_count += 1
        #     if len(p['path']) <= hop_constraint and p['DR'] >= DR_constraint:
        #         best_DR, best_hop = best_paths[gu.node_number]
        #         filtered_path_count += 1
        #         if p['DR'] == best_DR:
        #             total_score += 1
        #         else:
        #             hop_difference = len(p['path']) - best_hop
        #             if hop_difference <= 0:
        #                 total_score += p['DR'] / best_DR
        #             else:
        #                 total_score += (p['DR'] / best_DR) / hop_difference
    # score = 0 if filtered_path_count == 0 else (total_score / filtered_path_count) * (filtered_path_count / total_path_count)
    
    # Lognam: try to have another BP calculation to encourage exploration
    # score = 0 if filtered_path_count == 0 else total_score 

    return total_score

import copy

def quantify_network_partitioning(ground_users, UAV_nodes, backhaul_connection, ratio, DRPenalty, BPHopConstraint, BPDRConstraint, UAVInfo, DRScore, BPScore, ratioDR, ratioBP, scene_info):
    if ratioDR + ratioBP != 1:
        raise ValueError("The sum of ratio must be 1.")

    all_dropped_situation = select_all_drops(backhaul_connection, ratio)
    avgDRScore = 0
    avgBPScore = 0
    all_dropped_situation_count = len(all_dropped_situation)

    for single_dropped_situation in all_dropped_situation:
        dropped_backhaul_connection = copy.deepcopy(backhaul_connection)
        for dropped_node in single_dropped_situation:
            dropped_backhaul_connection = remove_node(dropped_backhaul_connection, dropped_node)

        curDRScore = quantify_data_rate(ground_users, dropped_backhaul_connection, DRPenalty, UAVInfo)
        curBPScore = quantify_backup_path(ground_users, UAV_nodes, dropped_backhaul_connection, BPHopConstraint, BPDRConstraint, scene_info)
        avgDRScore += curDRScore
        avgBPScore += curBPScore

    if all_dropped_situation_count == 0:
        curDRScore = quantify_data_rate(ground_users, backhaul_connection, DRPenalty, UAVInfo)
        curBPScore = quantify_backup_path(ground_users, UAV_nodes, backhaul_connection, BPHopConstraint, BPDRConstraint, scene_info)
    else:
        avgDRScore /= all_dropped_situation_count
        avgBPScore /= all_dropped_situation_count

    score = 0 if DRScore == 0 else ratioDR * (avgDRScore / DRScore)
    score += 0 if BPScore == 0 else ratioBP * (avgBPScore / BPScore)
    return score

def select_all_drops(backhaul_connection, ratio):
    UAV_num = len(backhaul_connection.allPaths)
    max_len = int((UAV_num) * ratio)
    
    elements = list(range(UAV_num))
    
    result = []
    for r in range(1, max_len + 1):
        result.extend(combinations(elements, r))
    
    return [list(comb) for comb in result] 

def remove_node(backhaul_connection, n):
    # UAVMapCopy = copy.deepcopy(backhaul_connection)
    # iterate all the keys (starter UAV node)
    for key in backhaul_connection.allPaths:
        # filter out all the paths that do not include node n
        backhaul_connection.allPaths[key] = [path_record for path_record in backhaul_connection.allPaths[key] if n not in path_record['path'] and path_record['path'][0] != n]
    return backhaul_connection

def integrate_quantification(value1, value2, value3, weight1, weight2, weight3):
    # make sure the sum of weighty is 1
    total_weight = weight1 + weight2 + weight3 
    if total_weight != 1:
        raise ValueError("The sum of weights must be 1.")
    
    # calculate the weighted sum
    integrated_value = value1 * weight1 + value2 * weight2 + value3 * weight3
    
    return integrated_value

import math

def disable_bs_edges_in_state(state, bs_index, num_uavs):

    state_list = list(state)

    num_nodes = int((1 + math.sqrt(1 + 8 * len(state))) / 2)
    
    bs_start_index = bs_index+num_uavs

    for i in range(num_nodes):
        if i == bs_start_index: continue
        edge_index = edge_index_in_state(i, bs_start_index, num_nodes)
        state_list[edge_index] = "0"
    
    return ''.join(state_list)

def edge_index_in_state(node1, node2, num_nodes):
    if node1 > node2:
        node1, node2 = node2, node1
    return int(node1 * (2 * num_nodes - node1 - 1) / 2 + (node2 - node1 - 1))

import time
def find_best_backhaul_topology(GU_nodes, UAV_nodes, BS_nodes, eps, reward_hyper, episodes, scene_info = None, print_prog = False, initialize_as_all_0 = False):
    best_state = ""
    q_table = {}
    reward_track = []
    RS_track = []
    max_reward = 0
    best_resilience_score = -math.inf
    best_backhaul_connection = None

    best_reward_track = []
    best_RS_track = []

    num_nodes = len(BS_nodes) + len(UAV_nodes)

    if initialize_as_all_0:
        state = '0' * int((num_nodes * (num_nodes - 1) / 2))
    else:
        state = '1' * int((num_nodes * (num_nodes - 1) / 2))

    # print("state is: "+str(state))
        
    # set an initial state for hard scene
    state = "111111100111111111111111101111111111111110111"
    
    start_time = time.time()

    # best_state_backhaul_connection = None

    epsilon = eps
    # epsilon = 1

    for episode in range(episodes):
        # print("Episode: "+str(episode))
        print("At episode "+str(episode)+", Q tables has explore: "+str(len(q_table))+" states.") if print_prog else None

        # if episode > 0:
        #     state = generate_random_binary_string(state)

        next_possible_states = generate_adjacent_states(state)
        states_scores, end_flag = process_states(next_possible_states, q_table, scene_info, GU_nodes, UAV_nodes, BS_nodes, reward_hyper)

        next_state, next_state_score = take_action(states_scores, epsilon)

        if print_prog:
            print("Next state is: "+str(next_state))

        if next_state_score[0] > max_reward:
            best_state = next_state
            max_reward = next_state_score[0]
            best_resilience_score = next_state_score[1]

            best_backhaul_connection = next_state_score[2]

            print("New topology with highest reward is found, new reward is: "+str(max_reward)+", at state: "+str(best_state)) if print_prog else None

        reward_track.append(next_state_score[0])
        RS_track.append(next_state_score[1])

        best_reward_track.append(max_reward)
        best_RS_track.append(best_resilience_score)

        if not end_flag:
            state = generate_random_binary_string(state)
        else:
            state = next_state

    if print_prog:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The code block ran in {elapsed_time} seconds")
        print(f"Best state: {best_state}")
        print(f"Max reward: {max_reward}")

    # set connections for UAVs and BSs
    set_connected_edges(best_state, UAV_nodes, BS_nodes)
    return best_state, max_reward, best_resilience_score, reward_track, RS_track, best_reward_track, best_RS_track, best_backhaul_connection