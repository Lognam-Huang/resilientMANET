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
    # print(UAV_overload)
            
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
        gini = sum_of_differences / (2 * n**2 * cumulative_loads)
        
        return gini
    
    overload_score = gini_coefficient(UAV_overload)
    
    # return any(value > overload_constraint for value in UAV_overload.values())
    return overload_score

def get_nodes_with_max_dr(data):
    # find path with max DR using lambda
    max_dr_path = max(data, key=lambda x: x['DR'])['path']

    # print(max_dr_path)
    return max_dr_path