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

from functions.scene_visualization import scene_visualization

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


# set position for stable result
ground_users[0].set_position((250,200,0))
ground_users[1].set_position((250,400,0))
ground_users[2].set_position((250,600,0))
ground_users[3].set_position((600,250,0))
ground_users[4].set_position((600,450,0))


UAVNodes[0].set_position((250,200,200))
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

# visualize scene
scene_visualization(ground_users, UAVNodes, ABSNodes, blocks, scene)