def quantify_data_rate(UAVMap, r):
    # print(UAVMap)
    # 1. 获取每个元素的最大DR
    max_data_rates = [max(paths, key=lambda x: x['DR'])['DR'] if paths else 0 for paths in UAVMap.allPaths.values()]
    
    print(max_data_rates)
    
    # 2. 计算所有元素的最小DR和平均DR
    min_DR = min(max_data_rates)
    avg_DR = sum(max_data_rates) / len(max_data_rates)
    
    # 3. 使用公式计算score
    score = r * min_DR + (1 - r) * avg_DR
    return score