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
                
                if onPathNode < numUAV:
                    UAV_overload[onPathNode] += curPathDR
        else:
            best_DRs[start] = None
    
    # UAV_overload = {}
    # print(best_DRs)
    # print(UAV_overload)
    
    return any(value > overload_constraint for value in UAV_overload.values())

def get_nodes_with_max_dr(data):
    # find path with max DR using lambda
    max_dr_path = max(data, key=lambda x: x['DR'])['path']
    return max_dr_path