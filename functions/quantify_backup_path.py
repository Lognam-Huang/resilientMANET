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
    max_path_count = max(len(paths) for paths in AllPaths.values())
    for start, paths in AllPaths.items():
        for p in paths:
            if hop_count(p['path']) <= hop_constraint and p['DR'] >= DR_constraint:
                if p['DR'] == best_DRs[start]:  # best path scores 1 point
                    total_score += 1
                else:
                    total_score += p['DR'] / best_DRs[start]

    # sum of the scores divided by the maximum number of path
    result = total_score / max_path_count
    return result