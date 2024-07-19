import copy
import sys
sys.path.append('../functions')
from functions.path_is_blocked import path_is_blocked
from functions.calculate_data_rate import calculate_data_rate

class UAVMap:
    def __init__(self, UAVs, ABSs, blocks, UAVInfo):
        self.UAVs = UAVs
        self.ABSs = ABSs
        self.blocks = blocks
        self.UAVInfo = UAVInfo
        
        
        numUAV = len(UAVs)
        
        self.allPaths = {i: [] for i in range(numUAV)}
        
        self.findAllPaths(UAVs, ABSs, blocks)
        
    def findAllPaths(self, UAVs, ABSs, blocks):
        numUAV = len(UAVs)
        numABS = len(ABSs)
        
        UAVCopy = copy.deepcopy(UAVs)
                
        for k in range (numABS):
            curABS = ABSs[k]
            for curConnectedUAV in curABS.connected_nodes:
                UAVCopy[curConnectedUAV].add_connection(numUAV+k)
        
        for i in range(numUAV):
            for j in range(numABS):
                # print("Current i:"+str(i))
                # print("Current j:"+str(numUAV+j))
                
                
                visitedNodes = [False]*numUAV
                
                self.findEachPath(i, numUAV+j, visitedNodes, [], numUAV, UAVCopy, float('inf'))
        
    
    def findEachPath(self, curNode, targetNode, visitedNodes, curPath, numUAV, UAVCopy, curDR):     
        curPath.append(curNode)
        if curNode == targetNode:
            finalPath = copy.deepcopy(curPath)
            
            self.allPaths[finalPath[0]].append({'path':finalPath, 'DR':curDR})
            return         
        elif curNode >= numUAV:
            return        
        else:
            
            visitedNodes[curNode] = True
            for connectedNode in UAVCopy[curNode].connected_nodes:
                if connectedNode < numUAV:
                    if visitedNodes[connectedNode] == True: continue                
                if connectedNode < numUAV:
                    isBlocked = path_is_blocked(self.blocks, UAVCopy[curNode], UAVCopy[connectedNode])
                    newDR = calculate_data_rate(self.UAVInfo, UAVCopy[curNode].position, UAVCopy[connectedNode].position, isBlocked)
                else:
                    isBlocked = path_is_blocked(self.blocks, UAVCopy[curNode], self.ABSs[connectedNode-numUAV])
                    newDR = calculate_data_rate(self.UAVInfo, UAVCopy[curNode].position, self.ABSs[connectedNode-numUAV].position, isBlocked)
                self.findEachPath(connectedNode, targetNode, visitedNodes, curPath, numUAV, UAVCopy, min(curDR, newDR))
                
                curPath.pop()
            
            visitedNodes[curNode] = False

    # String representation
    def __str__(self):
        return f"UAVMap(allPaths={self.allPaths})"

def find_best_paths_to_bs(UAVMap):
    best_paths = {}

    for uav, paths in UAVMap.allPaths.items():
        if uav > -1 and len(paths)>0:
            best_path = max(paths, key=lambda x: x['DR'])
            best_paths[uav] = best_path['path']
    
    return best_paths

# def extract_gu_to_uav_connections(ground_users):
#     gu_to_uav = {}

#     for gu_index, user in enumerate(ground_users):
#         gu_to_uav[gu_index] = user.connected_nodes

#     return gu_to_uav