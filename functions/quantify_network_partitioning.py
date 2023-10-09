import random
import copy
from itertools import combinations

from functions.quantify_data_rate import quantify_data_rate
from functions.quantify_backup_path import quantify_backup_path

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
    
    score = ratioDR*(avgDRScore/DRScore) + ratioBP*(avgBPScore/BPScore)
    return score


def remove_node(UAVMap, n):
    UAVMapCopy = copy.deepcopy(UAVMap)
    # iterate all the keys (starter UAV node)
    for key in UAVMapCopy.allPaths:
        # filter out all the paths that do not include node n
        UAVMapCopy.allPaths[key] = [path_record for path_record in UAVMapCopy.allPaths[key] if n not in path_record['path'] and path_record['path'][0] != n]
    return UAVMapCopy

def select_drop(UAVMap, ratio):
    numUAV = len(UAVMap.allPaths)
    num_samples = int(numUAV * ratio)  
    return random.sample(range(numUAV), num_samples)
    


def select_all_drops(UAVMap, ratio):
    numUAV = len(UAVMap.allPaths)
    max_len = int((numUAV) * ratio)
    
    elements = list(range(numUAV))
    
    result = []
    for r in range(1, max_len + 1):
        result.extend(combinations(elements, r))
    
    return [list(comb) for comb in result] 
    # return 1