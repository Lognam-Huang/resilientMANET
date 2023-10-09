import json

from classes.Nodes import Nodes
from functions.generate_users import generate_users
from functions.generate_UAVs import generate_UAVs
from functions.calculate_data_rate import calculate_data_rate
from functions.dB_conversion import dB_conversion
from functions.data_unit_conversion import data_unit_conversion
from classes.UAVMap import UAVMap
from functions.path_is_blocked import path_is_blocked
from functions.quantify_data_rate import quantify_data_rate
from functions.quantify_backup_path import quantify_backup_path
from functions.quantify_network_partitioning import quantify_network_partitioning
from functions.integrate_quantification import integrate_quantification
from functions.measure_overload import measure_overload
from functions.print_nodes import print_nodes

from DQN import *
import matplotlib.pyplot as plt


from functions.quantify_network_partitioning import remove_node, select_drop

# read scenario data
with open('scene_data.json', 'r') as f:
    ini = json.load(f)

groundBaseStation = ini['baseStation']
blocks = ini['blocks']
UAVInfo = ini['UAV']
scene = ini['scenario']

# Generate ground users
ground_users = generate_users(5, blocks, scene['xLength'], scene['yLength'])

# for user in ground_users:
#     print(user)

# Generate random UAVs
defaultHeightUAV = 200
# UAVNodes = generate_UAVs(5, blocks, scene['xLength'], scene['yLength'], defaultHeightUAV, 10, 'basic UAV')
UAVNodes = generate_UAVs(3, blocks, scene['xLength'], scene['yLength'], defaultHeightUAV, 10, 'basic UAV')
# for node in UAVNodes:
#     print(node)


# Generate air base station
defaultHeightABS = 500
# ABSNodes = generate_UAVs(2, blocks, scene['xLength'], scene['yLength'], defaultHeightABS, 10, 'Air Base Station')
ABSNodes = generate_UAVs(1, blocks, scene['xLength'], scene['yLength'], defaultHeightABS, 10, 'Air Base Station')
# for node in ABSNodes:
#     print(node)

fakeGroundUser = [100, 200, 0]
fakeUAV = [100, 200, 150]

# testDataRateNoBlocks = calculate_data_rate(UAVInfo['bandwidth'], dB_conversion(UAVInfo['Power'], 'dBm')/1000, UAVInfo['TransmittingAntennaGain'], UAVInfo['ReceievingAntennaGain'], UAVInfo['CarrierFrequency'], fakeUAV, fakeGroundUser, False)
testDataRateNoBlocks = calculate_data_rate(UAVInfo, fakeUAV, fakeGroundUser, False)
# print("No block")
[convertedDataRateNoBlocks, convertedUnitNoBlocks] = data_unit_conversion(testDataRateNoBlocks, 'bps')
# print(str(convertedDataRateNoBlocks)+" "+convertedUnitNoBlocks)

# testDataRateWithBlocks = calculate_data_rate(UAVInfo['bandwidth'], dB_conversion(UAVInfo['Power'], 'dBm')/1000, UAVInfo['TransmittingAntennaGain'], UAVInfo['ReceievingAntennaGain'], UAVInfo['CarrierFrequency'], fakeUAV, fakeGroundUser, True)
testDataRateWithBlocks = calculate_data_rate(UAVInfo, fakeUAV, fakeGroundUser, True)
# print("With block")
[convertedDataRateWithBlocks, convertedUnitWithBlocks] = data_unit_conversion(testDataRateWithBlocks, 'bps')
# print(str(convertedDataRateWithBlocks)+" "+convertedUnitWithBlocks)

# print(UAVNodes[0].position)

# UAVNodes[0].set_position((340,490,50))
# UAVNodes[1].set_position((440,390,50))

# print(UAVNodes[0].position)

# # print(UAVNodes[0].position)
# print(path_is_blocked(blocks, UAVNodes[0], UAVNodes[1]))

# UAVNodes[0].set_position((0,0,0))
# UAVNodes[1].set_position((500,500,500))

# # print(UAVNodes[0].position)
# print(path_is_blocked(blocks, UAVNodes[0], UAVNodes[1]))

# Find all available paths in MANET topology

# UAVNodes[0].set_position((2,3,4))
# UAVNodes[0].set_connection([1,5])
# print(UAVNodes[0])

# NOTICE THAT ALL NODES START WITH INDEX 0

# UAVNodes[0].set_connection([1,2,3])
# UAVNodes[1].set_connection([0])
# UAVNodes[2].set_connection([0])
# UAVNodes[3].set_connection([0])

# ABSNodes[0].set_connection([3])
# ABSNodes[1].set_connection([2,4])

UAVNodes[0].set_connection([1,2])
UAVNodes[1].set_connection([0])
UAVNodes[2].set_connection([0])

ABSNodes[0].set_connection([0,2])

UAVMap = UAVMap(UAVNodes, ABSNodes, blocks, UAVInfo)
print(UAVMap)

# for node in UAVNodes:
#     print(node)

# quantify resilience score: data rate
DRPenalty = 0.5
DRScore = quantify_data_rate(UAVMap, DRPenalty, UAVInfo)
print("Data rate score of current topology:")
print(DRScore)

# for node in UAVNodes:
#     print(node)

# quantify resilience score: backup path 
BPHopConstraint = 4
BPDRConstraint = 100000000
BPScore = quantify_backup_path(UAVMap, BPHopConstraint, BPDRConstraint)
print("Backup path score of current topology:")
print(BPScore)

# quantify resilience score: network partitioning
# for node in UAVNodes:
#     print(node)
# newUAVMap = remove_node(UAVMap, 2)
# print(newUAVMap)

droppedRatio = 0.2
ratioDR = 0.6
ratioBP = 0.4
NPScore = quantify_network_partitioning(UAVMap, droppedRatio, DRPenalty, BPHopConstraint, BPDRConstraint, UAVInfo, DRScore, BPScore, ratioDR, ratioBP)
print("Network partitioning score of current topology:")
print(NPScore)

# integrate quantificaiton
weightDR = 0.2
weightBP = 0.5
weightNP = 0.3
ResilienceScore = integrate_quantification(DRScore, BPScore, NPScore, weightDR, weightBP, weightNP)
print("Resilience score is:")
print(ResilienceScore) 

# measure the overload of current MANET topology
overloadConstraint = 10000
OverloadScore = measure_overload(UAVMap, BPHopConstraint, BPDRConstraint, overloadConstraint)
print(OverloadScore)

# try to use RL to improve resilience score

# first try: Deep Q-Learning
# result is not satisfied
# one possible explanation is  that: as a single-agent, this method is too inefficient (step()) 
# meanwhile this method does not fit the scenario well
print("Now pass to DQN")
env = UAVEnvironment(UAVNodes, ABSNodes, blocks, UAVInfo)
state_size = len(env.UAVNodes) * 3
action_size = len(env.UAVNodes) * 5  # For each UAV: move in x, move in y, or don't move
agent = DQNAgent(state_size, action_size)
episodes = 100

rs_values = []

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        
        rs_current = env.get_RS()
        rs_values.append(rs_current)
        
        if done:
            print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
            break
    if len(agent.memory) > 32:
        agent.replay(32)
        
plt.plot(rs_values)
plt.xlabel('Episode')
plt.ylabel('RS Value')
plt.title('RS Value over Episodes')
plt.show()

# test
# print(data['scenario']['xLength'])  # 500
# print(data['baseStation']['bottomCorner'])  # [0, 0]

# print(groundBaseStation)
# print(blocks)
# print(UAVInfo)
# print(UAVInfo['bandwidth'])
# print(scene)

# Usage example
# node = Nodes([1, 2, 3], "basicUAV", 100)
# print(node)

# node.set_position([4, 5, 6])
# node.set_connection([1, 2, 3])
# print(node)

# testDrop = select_drop(UAVMap, 0.4)
# print(testDrop)

# print(UAVMap)
# print(newUAVMap)
# from test import test, test1
# test1()