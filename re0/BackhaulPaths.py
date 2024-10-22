import copy
import sys
import os
# sys.path.append('../functions')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functions.path_is_blocked import path_is_blocked
from functions.calculate_data_rate import calculate_data_rate

class BackhaulPaths:
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
                
        for k in range (numABS):
            curABS = ABSs[k]
            for curConnectedUAV in curABS.connected_nodes:
                UAVs[curConnectedUAV].add_connection(numUAV+k)
        
        for i in range(numUAV):
            for j in range(numABS):
                visitedNodes = [False]*numUAV
                
                self.findEachPath(i, numUAV+j, visitedNodes, [], numUAV, UAVs, float('inf'))
        
    
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
        return f"{self.allPaths}"
