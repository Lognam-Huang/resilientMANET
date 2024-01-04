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
        
        
    # @property
    # def allPaths(self, UAVs, ABSs, blocks):
    #     return findAllPaths(self, UAVs, ABSs, blocks)
    
    # @allPaths.setter
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
        
        # print(self.blocks)
        
        
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
                # print("Cur connected node:"+str(connectedNode))
                if connectedNode < numUAV:
                    if visitedNodes[connectedNode] == True: continue
                #     visitedNodes[connectedNode] = True
                # else:
                # print("Move to next node:"+str(connectedNode))
                # print(visitedNodes)
                # return self.findEachPath(connectedNode, targetNode, visitedNodes, curPath, numUAV, UAVCopy)
                
                if connectedNode < numUAV:
                    isBlocked = path_is_blocked(self.blocks, UAVCopy[curNode], UAVCopy[connectedNode])
                    newDR = calculate_data_rate(self.UAVInfo, UAVCopy[curNode].position, UAVCopy[connectedNode].position, isBlocked)
                    # print(UAVCopy[connectedNode].position)
                else:
                    isBlocked = path_is_blocked(self.blocks, UAVCopy[curNode], self.ABSs[connectedNode-numUAV])
                    newDR = calculate_data_rate(self.UAVInfo, UAVCopy[curNode].position, self.ABSs[connectedNode-numUAV].position, isBlocked)
                    # print(self.ABSs[connectedNode-numUAV].position)
                
                # # isBlocked = path_is_blocked(self.blocks, UAVCopy[curNode].position, UAVCopy[connectedNode].position)
                # print(isBlocked)
                
                # print(newDR)
                
                # print(UAVCopy[curNode].position)
                # print(UAVCopy[connectedNode])
                # print(connectedNode)
                
                
                # print(UAVCopy[curNode].position)
                # print(connectedNode)
                self.findEachPath(connectedNode, targetNode, visitedNodes, curPath, numUAV, UAVCopy, min(curDR, newDR))
                
                curPath.pop()
            
            visitedNodes[curNode] = False
        # return False

    # String representation
    def __str__(self):
        return f"UAVMap(allPaths={self.allPaths})"

