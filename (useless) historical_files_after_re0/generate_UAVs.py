import random
import sys
sys.path.append('../classes')
from classes.Nodes import Nodes

def generate_UAVs(UAVNumber, blocks, groundX, groundY, UAVHeight, edgeMin, nodeType):
    numNodes = UAVNumber
    userNodes = []
    nodeCount = 0

    while nodeCount < UAVNumber:
        x = random.random() * groundX
        y = random.random() * groundY
        z = UAVHeight

        nodeStatus = 0

        for block in blocks:
            squareXLength = block['size'][0]
            squareYLength = block['size'][1]
            squareZLength = block['height']
            squareBottomCorner = block['bottomCorner']

            if (x >= squareBottomCorner[0] and x <= squareBottomCorner[0] + squareXLength and
                y >= squareBottomCorner[1] and y <= squareBottomCorner[1] + squareYLength and
                squareZLength >= UAVHeight):
                
                nodeStatus = 1
                break

        if nodeStatus == 0:
            for i in range(numNodes):
                if i < len(userNodes):
                    currentNode = userNodes[i]
                    if (currentNode[0] - edgeMin <= x <= currentNode[0] + edgeMin and
                        currentNode[1] - edgeMin <= y <= currentNode[1] + edgeMin and
                        currentNode[2] - edgeMin <= z <= currentNode[2] + edgeMin):
                        
                        nodeStatus = 1
                        break

        if nodeStatus == 0:
            nodeCount += 1
            userNodes.append((x, y, UAVHeight))
            # userNodes.append([x, y, UAVHeight])

    UAVNodes = []

    node_number = 0

    for i in range(len(userNodes)):
        UAVNodes.append(Nodes(userNodes[i], nodeType, 0, node_number))
        node_number += 1

    return UAVNodes