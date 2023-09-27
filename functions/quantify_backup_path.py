def quantify_backup_path(UAVMap, hop_constraint, DR_constraint):
    AllPaths = UAVMap.allPaths
    # print(UAVMap)
    # print(AllPaths)
    
    # internal function, which is used to calculate the hop
    def hop_count(path):
        return len(path)

    # calculate the optimal Data Rate for each starting point
    best_DRs = {}
    for start, paths in AllPaths.items():
        filtered_paths = [p for p in paths if hop_count(p['path']) <= hop_constraint and p['DR'] >= DR_constraint]
        if filtered_paths:
            best_DRs[start] = max(p['DR'] for p in filtered_paths)
        else:
            best_DRs[start] = None
        # print(filtered_paths)
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
                if p['DR'] == best_DRs[start]:  # best path scores 1 point
                    total_score += 1
                    cur_node_score += 1
                else:
                    total_score += p['DR'] / best_DRs[start]
                    cur_node_score += p['DR'] / best_DRs[start]
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
    score = total_score / max_path_count
    
    normScore = norm(score, min_score, max_score)
    # result = total_score 
    return normScore

def norm(score, min_score, max_score):
    # making use of min-max normalization
    normScore = (score-max_score)/(score-min_score)
    return normScore