import random
import copy

# from quantify_data_rate import quantify_data_rate
# from quantify_backup_path import quantify_backup_path

def quantify_network_partitioning(UAVMap, ratio):
    score = 4396
    
    droppedNode = select_drop(UAVMap, ratio)
    print(droppedNode)
    
    for curNode in droppedNode:
        print(curNode)
        
    
    
    return score


def remove_node(UAVMap, n):
    UAVMapCopy = copy.deepcopy(UAVMap)
    # 遍历所有的键
    for key in UAVMapCopy.allPaths:
        # 使用列表推导式过滤出不包含节点n的路径
        UAVMapCopy.allPaths[key] = [path_record for path_record in UAVMapCopy.allPaths[key] if n not in path_record['path'] and path_record['path'][0] != n]
    return UAVMapCopy

def select_drop(UAVMap, ratio):
    numUAV = len(UAVMap.allPaths)
    num_samples = int(numUAV * ratio)  
    return random.sample(range(numUAV+1), num_samples)
    
    