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
    
    norScore = norm(score, UAVInfo)
    
    return norScore

def norm(score, UAVInfo):
    # print(UAVInfo['bandwidth'])
    normScore = score/UAVInfo['bandwidth']
    return normScore