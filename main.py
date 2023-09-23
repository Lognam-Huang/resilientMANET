import json

from classes.Nodes import Nodes
from functions.generate_users import generate_users
from functions.generate_UAVs import generate_UAVs
from functions.calculate_data_rate import calculate_data_rate
from functions.dB_conversion import dB_conversion
from functions.data_unit_conversion import data_unit_conversion
from classes.UAVMap import UAVMap
from functions.path_is_blocked import path_is_blocked

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
UAVNodes = generate_UAVs(5, blocks, scene['xLength'], scene['yLength'], defaultHeightUAV, 10, 'basic UAV')
# for node in UAVNodes:
#     print(node)


# Generate air base station
defaultHeightABS = 500
ABSNodes = generate_UAVs(2, blocks, scene['xLength'], scene['yLength'], defaultHeightABS, 10, 'Air Base Station')
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

print(UAVNodes[0].position)

UAVNodes[0].set_position((340,490,50))
UAVNodes[1].set_position((440,390,50))

print(UAVNodes[0].position)

# print(UAVNodes[0].position)
print(path_is_blocked(blocks, UAVNodes[0], UAVNodes[1]))

UAVNodes[0].set_position((0,0,0))
UAVNodes[1].set_position((500,500,500))

# print(UAVNodes[0].position)
print(path_is_blocked(blocks, UAVNodes[0], UAVNodes[1]))

# Find all available paths in MANET topology

# UAVNodes[0].set_position((2,3,4))
# UAVNodes[0].set_connection([1,5])
# print(UAVNodes[0])

# NOTICE THAT ALL NODES START WITH INDEX 0

UAVNodes[0].set_connection([1,2,6])
UAVNodes[1].set_connection([0])
UAVNodes[2].set_connection([0])

ABSNodes[0].set_connection([3])
ABSNodes[1].set_connection([2,4])

# for node in UAVNodes:
#     print(node)

UAVMap = UAVMap(UAVNodes, ABSNodes, blocks, UAVInfo)
print(UAVMap)

# for node in UAVNodes:
#     print(node)

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