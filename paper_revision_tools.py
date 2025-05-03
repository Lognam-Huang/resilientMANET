import json

# Load scene data from JSON file
# with open('scene_data_hard.json', 'r') as file:
# with open('scene_data_simple.json', 'r') as file:
# with open('scene_data.json', 'r') as file:
# with open('scene_data_mid.json', 'r') as file:
with open('scene_data_mid_dense.json', 'r') as file:
    scene_data = json.load(file)

blocks = scene_data['blocks']
scene = scene_data['scenario']
UAVInfo = scene_data['UAV']
baseStation = scene_data['baseStation']
nodeNumber = scene_data['nodeNumber']

num_GU = nodeNumber['GU']
num_UAV = nodeNumber['UAV']
num_BS = len(scene_data['baseStation'])

# ---------
from node_functions import generate_nodes, print_node, move_ground_users, get_nodes_position, add_or_remove_gu
# 0-GU, 1-UAV, 2-BS
ground_users = generate_nodes(num_GU, 0)
UAV_nodes = generate_nodes(num_UAV, 1)
BS_nodes = generate_nodes(num_BS, 2)

for i in range(num_BS):
    BS_nodes[i].set_position((baseStation[i]['bottomCorner'][0], baseStation[i]['bottomCorner'][1], baseStation[i]['height'][0]))

# # print_node(ground_users)

GU_position_TD = []
# max_movement_distance = 5
max_movement_distance = 30

# print(scene_data)
# print("--")
# print(scene_data["blocks"])

from visualization_functions import scene_visualization
scene_visualization(ground_users=None, UAV_nodes=None, air_base_station=None, scene_info=scene_data, show_axes_labels=False)

for i in range(10):
    move_ground_users(ground_users, blocks, scene['xLength'], scene['yLength'], 2000)

for i in range(50):
    # add_or_remove_gu(ground_users)
    move_ground_users(ground_users, blocks, scene['xLength'], scene['yLength'], max_movement_distance)
    # print(get_nodes_position(ground_users))

    GU_position_TD.append(get_nodes_position(ground_users))

    # scene_visualization(ground_users, UAV_nodes=None, air_base_station=None, scene_info=scene_data)

import pandas as pd
df = pd.DataFrame(GU_position_TD)

csv_file = "ground_user_positions_for_mid_scene_50_stable_dense.csv"
df.to_csv(csv_file, index=False)

ground_users_positions = pd.read_csv(csv_file)

print(ground_users_positions.iloc[0])
print(len(ground_users_positions.iloc[0]))
print(ground_users_positions.iloc[0,0])
