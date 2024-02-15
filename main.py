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
from functions.quantify_user_rate import quantify_user_rate

from functions.get_3D_heatmap import get_3D_heatmap

from functions.scene_visualization import scene_visualization, visualize_2D_heatmap_per_layer

# from DQN import *
# import matplotlib.pyplot as plt


from functions.quantify_network_partitioning import remove_node, select_drop

# read scenario data
with open('scene_data.json', 'r') as f:
    ini = json.load(f)

groundBaseStation = ini['baseStation']
blocks = ini['blocks']
UAVInfo = ini['UAV']
scene = ini['scenario']

# print(blocks)
# print(scene)

# Generate ground users
ground_users = generate_users(5, blocks, scene['xLength'], scene['yLength'])

for user in ground_users:
    print(user)

# Generate random UAVs
defaultHeightUAV = 200
# UAVNodes = generate_UAVs(5, blocks, scene['xLength'], scene['yLength'], defaultHeightUAV, 10, 'basic UAV')
UAVNodes = generate_UAVs(3, blocks, scene['xLength'], scene['yLength'], defaultHeightUAV, 10, 'basic UAV')
for node in UAVNodes:
    print(node)


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

# set position for stable result
ground_users[0].set_position((250,200,0))
ground_users[1].set_position((250,400,0))
ground_users[2].set_position((250,600,0))
ground_users[3].set_position((600,250,0))
ground_users[4].set_position((600,450,0))


UAVNodes[0].set_position((250, 200, 200))
UAVNodes[1].set_position((250,600,200))
UAVNodes[2].set_position((600,350,200))

ABSNodes[0].set_position((440,390,500))

ground_users[0].set_connection([0])
ground_users[1].set_connection([1])
ground_users[2].set_connection([1])
ground_users[3].set_connection([2])
ground_users[4].set_connection([2])

UAVNodes[0].set_connection([1,2])
UAVNodes[1].set_connection([0])
UAVNodes[2].set_connection([0])

ABSNodes[0].set_connection([0,2])



UAVMap = UAVMap(UAVNodes, ABSNodes, blocks, UAVInfo)
print(UAVMap)

# # for node in UAVNodes:
# #     print(node)

# # quantify resilience score: user testDataRateNoBlocks
# URPenalty = 0.5
# URScore = quantify_user_rate(UAVMap, ground_users, blocks, UAVInfo, UAVNodes, URPenalty)
# print("User rate:")
# print(URScore)
# # for user in ground_users:
# #     print(user)

# # quantify resilience score: data rate
# DRPenalty = 0.5
# DRScore = quantify_data_rate(UAVMap, DRPenalty, UAVInfo)
# print("Data rate score of current topology:")
# print(DRScore)

# # for node in UAVNodes:
# #     print(node)

# # quantify resilience score: backup path 
# BPHopConstraint = 4
# BPDRConstraint = 100000000
# BPScore = quantify_backup_path(UAVMap, BPHopConstraint, BPDRConstraint)
# print("Backup path score of current topology:")
# print(BPScore)

# # quantify resilience score: network partitioning
# # for node in UAVNodes:
# #     print(node)
# # newUAVMap = remove_node(UAVMap, 2)
# # print(newUAVMap)

# droppedRatio = 0.2
# ratioDR = 0.6
# ratioBP = 0.4
# NPScore = quantify_network_partitioning(UAVMap, droppedRatio, DRPenalty, BPHopConstraint, BPDRConstraint, UAVInfo, DRScore, BPScore, ratioDR, ratioBP)
# print("Network partitioning score of current topology:")
# print(NPScore)

# # integrate quantificaiton
# weightUR = 0.3
# weightDR = 0.2
# weightBP = 0.4
# weightNP = 0.1
# ResilienceScore = integrate_quantification(URScore, DRScore, BPScore, NPScore, weightUR, weightDR, weightBP, weightNP)
# print("Resilience score is:")
# print(ResilienceScore) 

# # measure the overload of current MANET topology
# overloadConstraint = 10000
# OverloadScore = measure_overload(UAVMap, BPHopConstraint, BPDRConstraint, overloadConstraint)
# print(OverloadScore)

# visualize scene
# scene_visualization(ground_users, UAVNodes, ABSNodes, blocks, scene)

min_height = 195  # 最小高度
max_height = 205  # 最大高度

heatmap = get_3D_heatmap(ground_users, blocks, scene, min_height, max_height)

visualize_2D_heatmap_per_layer(heatmap=heatmap, min_height=min_height, max_height= max_height)

print("aa")
