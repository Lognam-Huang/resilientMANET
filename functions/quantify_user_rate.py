from functions.calculate_data_rate import calculate_data_rate
from functions.path_is_blocked import path_is_blocked

def quantify_user_rate(UAVMap, ground_users, blocks, UAVInfo, UAVNodes, r):
    # measure the quality of each user node in the MANET
    # for each user node, find the maximum, which means the corresponding UAV node
    
    # 1. get the maximum data rate for each UAV node
    max_data_rates = [max(paths, key=lambda x: x['DR'])['DR'] if paths else 0 for paths in UAVMap.allPaths.values()]
    # print(max_data_rates)
    
    # 2. get maximum DR for each GU corresponding UAV node
    DR_for_each_user = []
    for user in ground_users:
        # print(user.connected_nodes)
        # here we assume for each GU, there is only 1 connected UAV
        DR_for_each_user.append(max_data_rates[user.connected_nodes[0]])
    
    print(DR_for_each_user)
    
    # 3. calculate the maximum possible DR between each GU and its UAV node, and update DR for each GU
    i = 0
    for user in ground_users:
        isBlocked = path_is_blocked(blocks, user, UAVNodes[user.connected_nodes[0]])
        newDR = calculate_data_rate(UAVInfo, user.position, UAVNodes[user.connected_nodes[0]].position, isBlocked)
        # print(user.position)
        # print(UAVNodes[user.connected_nodes[0]].position)
        # DR_for_each_user[i] = min(DR_for_each_user[i], newDR)
        i += 1
                    
    min_DR = min(DR_for_each_user)
    avg_DR = sum(DR_for_each_user) / len(DR_for_each_user)
    
    # 3. calculate score based on equation
    score = r * min_DR + (1 - r) * avg_DR
    
    norScore = norm(score, UAVInfo)
    
    return norScore
    # return 1


def norm(score, UAVInfo):
    # print(UAVInfo['bandwidth'])
    normScore = score/UAVInfo['bandwidth']
    return normScore