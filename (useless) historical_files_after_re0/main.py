import json

from classes.Nodes import Nodes
from functions.generate_users import generate_users
from functions.generate_UAVs import generate_UAVs
from functions.calculate_data_rate import calculate_data_rate
from functions.dB_conversion import dB_conversion
from functions.data_unit_conversion import data_unit_conversion
from classes.UAVMap import UAVMap, find_best_paths_to_bs
from functions.path_is_blocked import path_is_blocked
from functions.quantify_data_rate import quantify_data_rate
from functions.quantify_backup_path import quantify_backup_path
from functions.quantify_network_partitioning import quantify_network_partitioning
from functions.integrate_quantification import integrate_quantification
from functions.measure_overload import measure_overload
from functions.print_nodes import print_nodes
from functions.quantify_user_rate import quantify_user_rate

from functions.get_3D_heatmap import get_3D_heatmap

from functions.scene_visualization import *

# from DQN import *
# import matplotlib.pyplot as plt


def extract_gu_to_uav_connections(ground_users):
    gu_to_uav = {}

    for gu_index, user in enumerate(ground_users):
        gu_to_uav[gu_index] = user.connected_nodes

    return gu_to_uav



from functions.quantify_network_partitioning import remove_node, select_drop

# # read scenario data, too slow for System Overview figure
# with open('scene_data.json', 'r') as f:
#     ini = json.load(f)
# with open('scene_data_simple.json', 'r') as file:
#     ini = json.load(file)
with open('scene_data_system_overview.json', 'r') as file:
    ini = json.load(file)

groundBaseStation = ini['baseStation']
blocks = ini['blocks']
UAVInfo = ini['UAV']
scene = ini['scenario']

# print(blocks)
# print(scene)

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

# NOTICE THAT ALL NODES START WITH INDEX 0

# UAVNodes[0].set_connection([1,2,3])
# UAVNodes[1].set_connection([0])
# UAVNodes[2].set_connection([0])
# UAVNodes[3].set_connection([0])

# ABSNodes[0].set_connection([3])
# ABSNodes[1].set_connection([2,4])

# set position for stable result
# ground_users[0].set_position((250,200,0))
# ground_users[1].set_position((250,400,0))
# ground_users[2].set_position((250,600,0))
# ground_users[3].set_position((600,250,0))
# ground_users[4].set_position((600,450,0))


# UAVNodes[0].set_position((250, 200, 200))
# UAVNodes[1].set_position((250,600,200))
# UAVNodes[2].set_position((600,350,200))

# ABSNodes[0].set_position((440,390,500))

# # ground_users[0].set_position((50,50,0))
# # ground_users[1].set_position((150,150,0))
# # ground_users[2].set_position((250,30,0))
# # ground_users[3].set_position((270,180,0))
# # ground_users[4].set_position((50,250,0))

# ground_users[0].set_position((50,50,0))
# ground_users[1].set_position((700,750,0))
# ground_users[2].set_position((250,700,0))
# ground_users[3].set_position((450,100,0))
# ground_users[4].set_position((780,250,0))

# # following position is set for simple scene

# UAVNodes[0].set_position((3, 12, 20))
# UAVNodes[1].set_position((18,10,13))
# UAVNodes[2].set_position((15,22,20))

# ABSNodes[0].set_position((15,20,40))

# ground_users[0].set_position((3,3,0))
# ground_users[1].set_position((5,20,0))
# ground_users[2].set_position((13,3,0))
# ground_users[3].set_position((20,15,0))
# ground_users[4].set_position((20,25,0))


# ground_users[0].set_connection([0])
# ground_users[1].set_connection([1])
# ground_users[2].set_connection([1])
# ground_users[3].set_connection([2])
# ground_users[4].set_connection([2])

# UAVNodes[0].set_connection([1,2])
# UAVNodes[1].set_connection([0])
# UAVNodes[2].set_connection([0])

# ABSNodes[0].set_connection([0,2])

# following position is set for system overview
UAVNodes[0].set_position((300, 650, 220))
UAVNodes[1].set_position((1000,400,230))
UAVNodes[2].set_position((600,1300,220))

ABSNodes[0].set_position((100, 1450,400))

ground_users[0].set_position((30,30,0))
ground_users[1].set_position((1100,900,0))
ground_users[2].set_position((1200,350,0))
ground_users[3].set_position((950,1300,0))
ground_users[4].set_position((1400,1400,0))


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

# print(UAVMap.allPaths)

best_paths_to_bs = find_best_paths_to_bs(UAVMap)
print(best_paths_to_bs)


gu_to_uav_connections = extract_gu_to_uav_connections(ground_users)
print(gu_to_uav_connections)

# for start, end in gu_to_uav_connections.items():
#     print(start)
#     print(end)

# visualize scene
# scene_visualization(ground_users=ground_users, UAV_nodes=UAVNodes, air_base_station=ABSNodes, blocks=blocks, scene_info=scene)
# scene_visualization(ground_users=ground_users, UAV_nodes=UAVNodes, air_base_station=ABSNodes, blocks=blocks, scene_info=scene, connection_GU_UAV=gu_to_uav_connections, connection_UAV_BS=best_paths_to_bs, line_alpha=0.5)

# min_height = 10  
# max_height = 20  

# heatmap = get_3D_heatmap(ground_users, blocks, scene, min_height, max_height)

# # visualize_2D_heatmap_per_layer(heatmap=heatmap, min_height=min_height, max_height= max_height)
# visualize_2D_heatmap_combined(heatmap=heatmap, min_height=min_height, max_height= max_height)

print("aa")
# print_nodes(UAVNodes)

# Try to draw figures for paper, system overview
# # 1: only scene and GU
# scene_visualization(ground_users=ground_users, air_base_station=ABSNodes, blocks=blocks, scene_info=scene, line_alpha=0.5, show_axes_labels=False)

# # 2: UAV positions are found
# scene_visualization(ground_users=ground_users, UAV_nodes=UAVNodes, air_base_station=ABSNodes, blocks=blocks, scene_info=scene, show_axes_labels=False)

# # 3: connectivity of UAV is found
# scene_visualization(ground_users=ground_users, UAV_nodes=UAVNodes, air_base_station=ABSNodes, blocks=blocks, scene_info=scene, connection_GU_UAV=gu_to_uav_connections, connection_UAV_BS=best_paths_to_bs, line_alpha=0.5, show_axes_labels=False)

# # 4: visualize heatmap, current UAV is at height 200
# min_height = 195
# max_height = 205  

# since we are using simple scene:
# min_height = 10
# max_height = 20  
# heatmap = get_3D_heatmap(ground_users, blocks, scene, min_height, max_height)

min_height = 220
max_height = 230

heatmap = get_3D_heatmap(ground_users, blocks, scene, min_height, max_height)

selected_heights = [220, 223, 227, 230]
# visualize_selected_2D_heatmaps(heatmap, selected_heights, min_height)
visualize_selected_heights_heatmaps(heatmap=heatmap, heights=selected_heights, min_height=min_height)

# print(heatmap)

# scene_visualization(ground_users=ground_users, air_base_station=ABSNodes, blocks=blocks, scene_info=scene, heatmap=heatmap)

# visualize_2D_heatmap_per_layer(heatmap=heatmap, min_height=min_height, max_height= max_height)
# visualize_2D_heatmap_combined(heatmap=heatmap, min_height=min_height, max_height= max_height)
# selected_heights = [10, 13, 16, 20]
# # visualize_selected_2D_heatmaps(heatmap, selected_heights, min_height)
# visualize_selected_heights_heatmaps(heatmap=heatmap, heights=selected_heights, min_height=min_height)