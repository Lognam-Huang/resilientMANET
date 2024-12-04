import json

# Load scene data from JSON file
# with open('scene_data_system_overview.json', 'r') as file:
# with open('scene_data_simple.json', 'r') as file:
# with open('scene_data.json', 'r') as file:
with open('scene_data_mid.json', 'r') as file:
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
from node_functions import generate_nodes, print_node
# 0-GU, 1-UAV, 2-BS
ground_users = generate_nodes(num_GU, 0)
UAV_nodes = generate_nodes(num_UAV, 1)


# Lognam: set simulation time
sim_time = 30

#

from node_functions import get_gu_to_uav_connections, move_gu_and_update_connections, get_nodes_position
from position_finding import find_optimal_uav_positions
from connectivity_finding import find_best_backhaul_topology, reward
from visualization_functions import scene_visualization, visualize_metrics
from node_functions import add_or_remove_gu, set_baseline_backhaul, get_gu_info_and_update_connections

# best_backhaul_connection = None

# # create a baseline backhaul for comparison
# baseline_UAV_nodes = generate_nodes(num_UAV, 1)
# baseline_BS_nodes = generate_nodes(num_BS, 2)

# baseline_BS_connections_for_mid_scene = [
#     [0,1,4],
#     [0,1,2,4]
# ]

# # for mid scene:
# baseline_UAV_positions_for_mid_scene = [(185, 100, 50), (126, 180, 50), (36, 147, 50), (36, 52, 50), (126, 19, 50)]

# baseline_UAV_connections_for_mid_scene = [
#     [2,4],
#     [2,4],
#     [0,1,3,4],
#     [2],
#     [0,1,2]
# ]

# baseline_BS_connections_for_simple_scene = [
#     [0,1]
# ]

# baseline_UAV_positions_for_simple_scene = [(3,3,10), (25,3,10), (2,25,10)]

# baseline_UAV_connections_for_simple_scene = [
#     [1,2],
#     [1],
#     [0],
# ]

# set_baseline_backhaul(baseline_UAV_nodes, baseline_UAV_positions_for_simple_scene, baseline_UAV_connections_for_simple_scene, baseline_BS_nodes, baseStation, baseline_BS_connections_for_simple_scene)
# # set_baseline_backhaul(baseline_UAV_nodes, baseline_UAV_positions_for_mid_scene, baseline_UAV_connections_for_mid_scene, baseline_BS_nodes, baseStation, baseline_BS_connections_for_mid_scene)

# # scene_visualization(ground_users, baseline_UAV_nodes, baseline_BS_nodes, scene_data, 0.3)

from connectivity_finding import get_backhaul_connection
# # baseline_state = "010111 010111 111101 001000 111011"
# # baseline_state = "010111 10111 1101 000 11 0"
# baseline_state_for_mid_scene = "010111101111101000110"
# baseline_state_for_simple_scene = "111010"

# # baseline_backhaul_connection = get_backhaul_connection(baseline_state_for_mid_scene, baseline_UAV_nodes, baseline_BS_nodes, scene_data)
# baseline_backhaul_connection = get_backhaul_connection(baseline_state_for_simple_scene, baseline_UAV_nodes, baseline_BS_nodes, scene_data)
# baseline_gu_capacity_TD = []
# baseline_uav_connections_TD = []

import pandas as pd

# ground_users_positions_simple_stable = pd.read_csv("ground_user_positions_for_simple_scene_50_stable.csv")
# ground_users_positions_mid_stable = pd.read_csv("ground_user_positions_for_mid_scene_50_stable.csv")

ground_users_positions_simple_dynamic = pd.read_csv("ground_user_positions_for_mid_scene_50_dynamic.csv")
# ground_users_positions_mid_stable = pd.read_csv("ground_user_positions_for_mid_scene_50_stable.csv")

# set first baseline
baseline_1_UAV_nodes = generate_nodes(num_UAV, 1)
baseline_1_BS_nodes = generate_nodes(num_BS, 2)

baseline_1_BS_connections_for_mid_scene = [
    [0,1,4],
    [0,1,2,4]
]

baseline_1_UAV_positions_for_mid_scene = [(50, 50, 50), (50, 150, 50), (100, 100, 50), (150, 50, 50), (150, 150, 50)]

baseline_1_UAV_connections_for_mid_scene = [
    [2,4],
    [2,4],
    [0,1,3,4],
    [2],
    [0,1,2]
]

# baseline_1_state_for_mid_scene = "010111101111101000110"
baseline_1_state_for_mid_scene = "000011000010011010110"

set_baseline_backhaul(baseline_1_UAV_nodes, baseline_1_UAV_positions_for_mid_scene, baseline_1_UAV_connections_for_mid_scene, baseline_1_BS_nodes, baseStation, baseline_1_BS_connections_for_mid_scene)
baseline_1_backhaul_connection = get_backhaul_connection(baseline_1_state_for_mid_scene, baseline_1_UAV_nodes, baseline_1_BS_nodes, scene_data)
baseline_1_gu_capacity_TD = []
baseline_1_uav_connections_TD = []

# set second baseline
baseline_2_UAV_nodes = generate_nodes(num_UAV, 1)
baseline_2_BS_nodes = generate_nodes(num_BS, 2)

baseline_2_BS_connections_for_mid_scene = [
    [0,1,4],
    [0,1,2,4]
]

baseline_2_UAV_positions_for_mid_scene = [(50, 50, 50), (50, 150, 50), (100, 100, 50), (150, 50, 50), (150, 150, 50)]

baseline_2_UAV_connections_for_mid_scene = [
    [2,4],
    [2,4],
    [0,1,3,4],
    [2],
    [0,1,2]
]


# baseline_2_state_for_mid_scene = "000011000010011010110"
baseline_2_state_for_mid_scene = "111111111111111111111"

set_baseline_backhaul(baseline_2_UAV_nodes, baseline_2_UAV_positions_for_mid_scene, baseline_2_UAV_connections_for_mid_scene, baseline_2_BS_nodes, baseStation, baseline_2_BS_connections_for_mid_scene)
baseline_2_backhaul_connection = get_backhaul_connection(baseline_2_state_for_mid_scene, baseline_2_UAV_nodes, baseline_2_BS_nodes, scene_data)
baseline_2_gu_capacity_TD = []
baseline_2_uav_connections_TD = []

# from functions.path_is_blocked import path_is_blocked
# from classes.Nodes import Nodes
# print(path_is_blocked(blocks, Nodes((50,50,50)), Nodes((90,10,0))))
# print(path_is_blocked(blocks, Nodes((50,150,50)), Nodes((90,10,0))))

for cur_time_frame in range(sim_time):  
    
    ground_users, gu_to_uav_connections_1, gu_to_bs_capacity_1 = get_gu_info_and_update_connections(ground_users_positions_simple_dynamic, cur_time_frame, blocks, UAV_nodes, UAVInfo, None)

    baseline_1_gu_to_uav_connections, baseline_1_gu_to_bs_capacity = get_gu_to_uav_connections(ground_users, baseline_1_UAV_nodes, UAVInfo, blocks, baseline_1_backhaul_connection)

    baseline_1_uav_connections_TD.append(baseline_1_gu_to_uav_connections)
    baseline_1_gu_capacity_TD.append(baseline_1_gu_to_bs_capacity)  


    # scene_visualization(ground_users, baseline_1_UAV_nodes, baseline_1_BS_nodes, scene_data, 0.3)

    baseline_state = cur_time_frame % 4
    
    if baseline_state == 0:
        baseline_2_UAV_positions_for_mid_scene = [(30, 30, 50), (30, 130, 50), (80, 80, 50), (130, 30, 50), (130, 130, 50)]
    elif baseline_state == 1:
        baseline_2_UAV_positions_for_mid_scene = [(70, 30, 50), (70, 130, 50), (120, 80, 50), (170, 30, 50), (170, 130, 50)]
    elif baseline_state == 2:
        baseline_2_UAV_positions_for_mid_scene = [(70, 70, 50), (70, 170, 50), (120, 120, 50), (170, 70, 50), (170, 170, 50)]
    elif baseline_state == 3:
        baseline_2_UAV_positions_for_mid_scene = [(30, 30, 50), (30, 130, 50), (80, 80, 50), (130, 30, 50), (130, 130, 50)]
    else:
        print("Error state for baseline 2.")

    set_baseline_backhaul(baseline_2_UAV_nodes, baseline_2_UAV_positions_for_mid_scene, baseline_2_UAV_connections_for_mid_scene, baseline_2_BS_nodes, baseStation, baseline_2_BS_connections_for_mid_scene)
    baseline_2_backhaul_connection = get_backhaul_connection(baseline_2_state_for_mid_scene, baseline_2_UAV_nodes, baseline_2_BS_nodes, scene_data)
    baseline_2_gu_to_uav_connections, baseline_2_gu_to_bs_capacity = get_gu_to_uav_connections(ground_users, baseline_2_UAV_nodes, UAVInfo, blocks, baseline_2_backhaul_connection)

    # scene_visualization(ground_users, baseline_2_UAV_nodes, baseline_2_BS_nodes, scene_data, 0.3)

    baseline_2_uav_connections_TD.append(baseline_2_gu_to_uav_connections)
    baseline_2_gu_capacity_TD.append(baseline_2_gu_to_bs_capacity)  


# record data
recorded_data = {
    # "GU Position": [tuple(pos) for pos in GU_position_TD], 
    # "UAV Position": [tuple(pos) for pos in UAV_position_TD],
    # "State": state_TD,
    "UAV Connections": baseline_1_uav_connections_TD,
    "GU Capacity": baseline_1_gu_capacity_TD,
}

recorded_df = pd.DataFrame(recorded_data)

recorded_df.to_csv("experiment_result_mid_baseline_1.csv", index=False)

recorded_data_2 = {
    # "GU Position": [tuple(pos) for pos in GU_position_TD], 
    # "UAV Position": [tuple(pos) for pos in UAV_position_TD],
    # "State": state_TD,
    "UAV Connections": baseline_2_uav_connections_TD,
    "GU Capacity": baseline_2_gu_capacity_TD,
}

recorded_df_2 = pd.DataFrame(recorded_data_2)

recorded_df_2.to_csv("experiment_result_mid_baseline_2.csv", index=False)

from visualization_functions import visualize_simulation, visualize_simulation_together, visualize_simulation_with_baseline
if sim_time > 0:

    # print(uav_connections_TD)
    # print(gu_capacity_TD)
    # # print(baseline_uav_connections_TD)
    # # print(baseline_gu_capacity_TD)
    # print(GU_position_TD)

    visualize_simulation(baseline_1_uav_connections_TD, baseline_1_gu_capacity_TD, num_UAV)
    visualize_simulation(baseline_2_uav_connections_TD, baseline_2_gu_capacity_TD, num_UAV)
    # visualize_simulation(uav_connections_TD, gu_capacity_TD, num_UAV)
    # visualize_simulation(baseline_uav_connections_TD, baseline_gu_capacity_TD, num_UAV)
    # visualize_simulation_with_baseline(uav_connections_TD, gu_capacity_TD, baseline_uav_connections_TD, baseline_gu_capacity_TD, num_UAV)
    # visualize_simulation_together(uav_connections_TD, gu_capacity_TD, num_UAV)
    # visualize_metrics(reward_TD, RS_TD)

        