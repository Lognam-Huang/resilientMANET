import json
from functions.generate_UAVs import generate_UAVs
from functions.print_nodes import print_nodes

from itertools import combinations

from classes.UAVMap import UAVMap
# from functions.quantify_data_rate import quantify_data_rate
# from functions.quantify_backup_path import quantify_backup_path
# from functions.quantify_network_partitioning import quantify_network_partitioning
# from functions.integrate_quantification import integrate_quantification

def aa():
    print("test")

# all quantification should only need:
    # state of connection 
    # UAV position
    # ABS position
    # block information
    # UAV information

# since we have already quantify RS using UAVMap, we now get UAVMap with certain information
    
    # necessary data
# read scenario data
with open('scene_data.json', 'r') as f:
    ini = json.load(f)

# groundBaseStation = ini['baseStation']
blocks = ini['blocks']
UAVInfo = ini['UAV']
scene = ini['scenario']
    
def get_UAVMap(state, UAV_position = 0, ABS_position = 0):
    # print(state)
    # print(UAV_position)
    # print(ABS_position)

    # print(len(UAV_position))
    num_UAV = len(UAV_position)
    num_ABS = len(ABS_position)

    # Generate random UAVs
    defaultHeightUAV = 200
    # UAVNodes = generate_UAVs(5, blocks, scene['xLength'], scene['yLength'], defaultHeightUAV, 10, 'basic UAV')
    UAVNodes = generate_UAVs(num_UAV, blocks, scene['xLength'], scene['yLength'], defaultHeightUAV, 10, 'basic UAV')

    # Generate air base station
    defaultHeightABS = 500
    # ABSNodes = generate_UAVs(2, blocks, scene['xLength'], scene['yLength'], defaultHeightABS, 10, 'Air Base Station')
    ABSNodes = generate_UAVs(num_ABS, blocks, scene['xLength'], scene['yLength'], defaultHeightABS, 10, 'Air Base Station')

    # the positions of UAV/ABS are generated randomly, which means we need to input them
    # according to Node, we need to make sure the position is in the format of tuple
    UAV_position = tuple(tuple(int(coord) for coord in triplet) for triplet in UAV_position)
    ABS_position = tuple(tuple(int(coord) for coord in triplet) for triplet in ABS_position)

    for i in range(0,num_UAV):
        UAVNodes[i].set_position(UAV_position[i])

    for i in range(0,num_ABS):
        ABSNodes[i].set_position(ABS_position[i])
    
    # print_nodes(UAVNodes, True)
    # print_nodes(ABSNodes, True)
        
    # we should set connection based on the state 
    get_connected_edges(''.join(str(int(i)) for i in state), UAVNodes, ABSNodes)
    # print(len(UAVNodes))

    # print_nodes(UAVNodes, False)
    # print_nodes(ABSNodes, False)

    NodeMap = UAVMap(UAVNodes, ABSNodes, blocks, UAVInfo)
    # print(NodeMap)

    return NodeMap

def get_connected_edges(edge_state, UAVNodes, ABSNodes):

    m = len(UAVNodes)
    n = len(ABSNodes)
    total_nodes = m + n
    edges = list(combinations(range(total_nodes), 2))
    connected_edges = [edges[i] for i, state in enumerate(edge_state) if state == '1']
    
    # Initialize connection lists for each node
    UAVConnections = {i: [] for i in range(m)}  # for m UAV nodes
    ABSConnections = {i: [] for i in range(n)}  # for n ABS nodes
    
    # Fill in the connections based on the connected edges
    for edge in connected_edges:
        a, b = edge
        
        # Both nodes are UAVs
        if a < m and b < m:
            UAVConnections[a].append(b)
            UAVConnections[b].append(a)
        # One node is UAV (a) and other is ABS (b)
        elif a < m and b >= m:
            ABSConnections[b - m].append(a)  # Keep a as is for UAV
        # One node is UAV (b) and other is ABS (a)
        elif b < m and a >= m:
            ABSConnections[a - m].append(b)  # Keep b as is for UAV
    
    # Now call the set_connection method for each UAV and ABS node
    for idx, connections in UAVConnections.items():
        UAVNodes[idx].set_connection(sorted(list(set(connections))))  # Remove duplicates and sort
    
    for idx, connections in ABSConnections.items():
        ABSNodes[idx].set_connection(sorted(list(set(connections))))  # Remove duplicates and sort


def quantify_data_rate(UAVMap, r, UAVInfo):
    # print(UAVMap)
    # 1. get the maximum data rate for each UAV node
    max_data_rates = [max(paths, key=lambda x: x['DR'])['DR'] if paths else 0 for paths in UAVMap.allPaths.values()]
    
    # print(max_data_rates)
    
    # 2. get the max and min data rate of all UAV node, based on highest DR of each node
    min_DR = min(max_data_rates)
    avg_DR = sum(max_data_rates) / len(max_data_rates)
    
    # 3. calculate score based on equation
    score = r * min_DR + (1 - r) * avg_DR

    def norm(score, UAVInfo):
        # print(UAVInfo['bandwidth'])
        normScore = score/UAVInfo['bandwidth']
        return normScore
    
    norScore = norm(score, UAVInfo)
    
    return norScore

def quantify_backup_path(UAVMap, hop_constraint, DR_constraint):
    AllPaths = UAVMap.allPaths
    # print(UAVMap)
    # print(AllPaths)
    
    # internal function, which is used to calculate the hop
    def hop_count(path):
        return len(path)-1

    # calculate the optimal Data Rate for each starting point
    # best_DRs = {}
    # best_hop_counts = {}
    
    best_paths = {}
    for start, paths in AllPaths.items():
        filtered_paths = [p for p in paths if hop_count(p['path']) <= hop_constraint and p['DR'] >= DR_constraint]
        if filtered_paths:
            # best_DRs[start] = max(p['DR'] for p in filtered_paths)
            best_path = max(filtered_paths, key=lambda p: p['DR'])
            # best_DRs[start] = best_path['DR']
            # best_hop_counts[start] = hop_count(best_path['path'])
            best_paths[start] = (best_path['DR'], hop_count(best_path['path']))
        
        else:
            # best_DRs[start] = None
            # best_hop_counts[start] = float('inf')
            
            best_paths[start] = (None, float('inf'))
        # print(filtered_paths)
        # print(best_DRs)
        # print(best_hop_counts)
    
    # print(best_DRs)

    # calculate the score for each path
    total_score = 0
    min_score = float('inf')
    max_score = -float('inf')
    cur_node_score = 0
    max_path_count = max(len(paths) for paths in AllPaths.values())
    
    for start, paths in AllPaths.items():
        for p in paths:
            if hop_count(p['path']) <= hop_constraint and p['DR'] >= DR_constraint:
                best_DR, best_hop = best_paths[start]
                
                if p['DR'] == best_DR:  # best path scores 1 point
                    total_score += 1
                    cur_node_score += 1
                else:
                    hop_difference = hop_count(p['path']) - best_hop
                    if hop_difference <= 0:
                        total_score += p['DR'] / best_DR
                        cur_node_score += p['DR'] / best_DR
                    else:
                        total_score += (p['DR'] / best_DR) / hop_difference
                        cur_node_score += (p['DR'] / best_DR) / hop_difference
        
            # print(p)
        # print(start)
        # print(paths)
        # print(cur_node_score)
        
        min_score = min(min_score, cur_node_score)
        max_score = max(max_score, cur_node_score)
        cur_node_score = 0

    # print("Min score:")
    # print(min_score)
    # print("Max score:")
    # print(max_score)
    
    # print("Total score:")
    # print(total_score)
    # print("Max path count:")
    # print(max_path_count)
    
    # sum of the scores divided by the maximum number of path
    score = 0 if max_path_count == 0 else total_score / max_path_count

    def norm(score, min_score, max_score):

        if score == 0: return 0

        # making use of min-max normalization
        normScore = (score-max_score)/(score-min_score)
        return normScore
    
    normScore = norm(score, min_score, max_score)
    # result = total_score 
    return normScore


import random
import copy
from itertools import combinations

# from functions.quantify_data_rate import quantify_data_rate
# from functions.quantify_backup_path import quantify_backup_path

def quantify_network_partitioning(UAVMap, ratio, DRPenalty, BPHopConstraint, BPDRConstraint, UAVInfo, DRScore, BPScore, ratioDR, ratioBP):
    score = 0
    if ratioDR+ratioBP != 1:
        raise ValueError("The sum of ratio must be 1.")
    
    # droppedNode = select_drop(UAVMap, ratio)
    # # print(droppedNode)
    
    # for curNode in droppedNode:
    #     # print(curNode)
    #     droppedUAVMap = remove_node(UAVMap, curNode)
    #     # print(droppedUAVMap)
        
    #     curDRScore = quantify_data_rate(droppedUAVMap, DRPenalty, UAVInfo)
    #     curBPScore = quantify_backup_path(droppedUAVMap, BPHopConstraint, BPDRConstraint)
        
        # print("Current DR score:")
        # print(curDRScore)
        # print("Current BP score:")
        # print(curBPScore)
    
    # print(select_all_drops(UAVMap, ratio))
    allDroppedNodes = select_all_drops(UAVMap, ratio)
    avgDRScore = 0
    avgBPScore = 0
    
    allDroppedSituation = len(allDroppedNodes)
    # print(allDroppedSituation)
    
    for curNodes in allDroppedNodes:
        droppedUAVMap = copy.deepcopy(UAVMap)
        for curNode in curNodes:
            # print(curNode)
            droppedUAVMap = remove_node(droppedUAVMap, curNode)
        
        curDRScore = quantify_data_rate(droppedUAVMap, DRPenalty, UAVInfo)
        curBPScore = quantify_backup_path(droppedUAVMap, BPHopConstraint, BPDRConstraint)
        
        # print("Current DR score:")
        # print(curDRScore)
        # print("Current BP score:")
        # print(curBPScore)
        avgDRScore += curDRScore
        avgBPScore += curBPScore
    
    if allDroppedSituation == 0:
        avgDRScore = 0
        avgBPScore = 0
    else:        
        avgDRScore /= allDroppedSituation
        avgBPScore /= allDroppedSituation
    
    # print("Average DR Score:")
    # print(avgDRScore)
    # print("Average BP Score:")
    # print(avgBPScore)
    
    # handle if DRScore or BPScore is 0
    score += 0 if DRScore == 0 else ratioDR*(avgDRScore/DRScore) 
    score += 0 if BPScore == 0 else ratioBP*(avgBPScore/BPScore)
    return score


def remove_node(UAVMap, n):
    UAVMapCopy = copy.deepcopy(UAVMap)
    # iterate all the keys (starter UAV node)
    for key in UAVMapCopy.allPaths:
        # filter out all the paths that do not include node n
        UAVMapCopy.allPaths[key] = [path_record for path_record in UAVMapCopy.allPaths[key] if n not in path_record['path'] and path_record['path'][0] != n]
    return UAVMapCopy

# def select_drop(UAVMap, ratio):
#     numUAV = len(UAVMap.allPaths)
#     num_samples = int(numUAV * ratio)  
#     return random.sample(range(numUAV), num_samples)
    


def select_all_drops(UAVMap, ratio):
    numUAV = len(UAVMap.allPaths)
    max_len = int((numUAV) * ratio)
    
    elements = list(range(numUAV))
    
    result = []
    for r in range(1, max_len + 1):
        result.extend(combinations(elements, r))
    
    return [list(comb) for comb in result] 
    # return 1


def integrate_quantification(value1, value2, value3, weight1, weight2, weight3):
    # make sure the sum of weighty is 1
    total_weight = weight1 + weight2 + weight3 
    if total_weight != 1:
        raise ValueError("The sum of weights must be 1.")
    
    # calculate the weighted sum
    integrated_value = value1 * weight1 + value2 * weight2 + value3 * weight3
    
    return integrated_value

def get_RS(UAVMap, DRPenalty, BPHopConstraint, BPDRConstraint, droppedRatio, ratioDR, ratioBP, weightDR, weightBP, weightNP):
    # quantify resilience score: data rate

    DRScore = quantify_data_rate(UAVMap, DRPenalty, UAVInfo)
    # print("Data rate score of current topology:")
    # print(DRScore)

    # quantify resilience score: backup path 
    BPScore = quantify_backup_path(UAVMap, BPHopConstraint, BPDRConstraint)
    # print("Backup path score of current topology:")
    # print(BPScore)

    
    NPScore = quantify_network_partitioning(UAVMap, droppedRatio, DRPenalty, BPHopConstraint, BPDRConstraint, UAVInfo, DRScore, BPScore, ratioDR, ratioBP)
    # print("Network partitioning score of current topology:")
    # print(NPScore)

    # integrate quantificaiton
    # weightUR = 0.3    
    ResilienceScore = integrate_quantification(DRScore, BPScore, NPScore, weightDR, weightBP, weightNP)
    # print("Resilience score is:")
    # print(ResilienceScore) 

    return ResilienceScore

def measure_overload(UAVMap, hop_constraint, DR_constraint, overload_constraint):
    AllPaths = UAVMap.allPaths
    # print(UAVMap)
    # print(AllPaths)
    
    # internal function, which is used to calculate the hop
    def hop_count(path):
        return len(path)

    # get the best/optimal path for each starter node
    numUAV = len(AllPaths)
    best_DRs = {}
    UAV_overload = {i: 0 for i in range(numUAV)}
    for start, paths in AllPaths.items():
        filtered_paths = [p for p in paths if hop_count(p['path']) <= hop_constraint and p['DR'] >= DR_constraint]
        if filtered_paths:
            # best_DRs[start] = max(p['DR'] for p in filtered_paths)
            curPathDR = max(p['DR'] for p in filtered_paths)
            
            for onPathNode in get_nodes_with_max_dr(filtered_paths):
                # print(onPathNode)
                
                # we only need to consider the overload of UAV nodes, no ABS nodes should be counted in this part  
                if onPathNode < numUAV:
                    UAV_overload[onPathNode] += curPathDR
        else:
            best_DRs[start] = None
    
    # UAV_overload = {}
    # print(best_DRs)
    print(UAV_overload)
            
    def gini_coefficient(uav_loads):
        """
        Calculate the Gini coefficient for UAV network loads.
        
        :param uav_loads: Dictionary of UAV node identifiers and their corresponding loads
        :return: Gini coefficient as a float
        """
        # Convert loads to a list of values and sort them
        loads = sorted(uav_loads.values())
        n = len(loads)
        cumulative_loads = sum(loads)
        
        # Calculate the Gini coefficient using the formula
        sum_of_differences = sum(abs(x - y) for x in loads for y in loads)
        gini = 0 if cumulative_loads == 0 else sum_of_differences / (2 * n**2 * cumulative_loads)
        
        return gini
    
    overload_score = gini_coefficient(UAV_overload)
    
    # return any(value > overload_constraint for value in UAV_overload.values())
    return overload_score

def get_nodes_with_max_dr(data):
    # find path with max DR using lambda
    max_dr_path = max(data, key=lambda x: x['DR'])['path']

    # print(max_dr_path)
    return max_dr_path

def d():
    print(1)