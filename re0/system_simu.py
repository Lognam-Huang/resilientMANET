import json
# from functions.print_nodes import print_nodes

# from gu_movement import move_ground_users, simulate_and_visualize_movements

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
BS_nodes = generate_nodes(num_BS, 2)

for i in range(num_BS):
    BS_nodes[i].set_position((baseStation[i]['bottomCorner'][0], baseStation[i]['bottomCorner'][1], baseStation[i]['height'][0]))

# from key_functions.uav_position_finding import *
position_params = {
    'weights': {
        'GU': 8,  # Weight for ground user connections
        'UAV': 2,  # Weight for UAV-to-UAV connections
        'BS': 1   # Weight for base station connections
    },
    'sparsity_parameter': 1  # Controls the density of the heatmap
    # ,
    # "eps" : 8,
    # "min_samples" : 2
}


# Lognam: find UAV connection
# from find_topo import *
reward_hyper = {
    'DRPenalty': 0.5,
    'BPHopConstraint': 4,
    'BPDRConstraint': 100000000,
    'droppedRatio': 0.2,
    'ratioDR': 0.6,
    'ratioBP': 0.4,
    'weightDR': 0.3,
    'weightBP': 0.4,
    'weightNP': 0.3,
    'overloadConstraint': 10000
}

# Q-learning hyperparameters
epsilon = 0.4
training_episodes= 200

# Lognam: set simulation time
sim_time = 10

# Lognam: try to switch scenes
max_movement_distance = 200

constraint_hyper = {
    'rewardConstraint': 0.8,
    'GUConstraint': 100000000,
    'max_uav_load_rate': 0.5
}

# rewardScore = 0
# OverloadScore = 0

reward_TD = []
RS_TD = []
OL_TD = []
gu_capacity_TD = []
UAV_capacity_TD = []
UAV_overload_TD = []

uav_connections_TD = []

min_gu_capacity_TD = []
# gu_capacity_at_a_moment = []

from node_functions import move_ground_users, get_gu_to_uav_connections, move_gu_and_update_connections

for i in range(10):
    move_gu_and_update_connections(ground_users, blocks, scene['xLength'], scene['yLength'], max_movement_distance, UAV_nodes, UAVInfo, None)
# print(move_gu_and_update_connections(ground_users, blocks, scene['xLength'], scene['yLength'], max_movement_distance, UAV_nodes, UAVInfo, None))
# move_ground_users(ground_users, blocks, scene['xLength'], scene['yLength'], max_movement_distance)

from position_finding import find_optimal_uav_positions, generate_3D_heatmap
from connectivity_finding import find_best_backhaul_topology, reward, set_connected_edges
from visualization_functions import scene_visualization, visualize_all_gu_capacity, visualize_metrics, visualize_all_min_gu_capacity, visualize_heatmap_slices, visualize_hierarchical_clustering, visualize_capacity_and_load, visualize_scores, visualize_best_scores, visualize_gu_by_connection

# print_node(UAV_nodes, -1, True)

# ----
# try to visualize for paper, section VII-B, C, D

# heatmap_for_visualization, best_position_for_visualization, max_connection_score_for_visualization, min_gu_bottleneck_for_visualization = generate_3D_heatmap(ground_users, scene_data, position_params['weights'], position_params['sparsity_parameter'])
# # print(heatmap_for_visualization)

# # selected_heights = [10, 15]
# selected_heights = [50, 55]
# visualize_heatmap_slices(heatmap_for_visualization, selected_heights, True)
# visualize_heatmap_slices(heatmap_for_visualization, selected_heights, False)

# # try to visualize GU, and hierarchichal clustering
# found_UAV_positions, hierachical_clustering_GU_records, gu_capacities_records, uav_load_records= find_optimal_uav_positions(
#             ground_users=ground_users, 
#             uavs=UAV_nodes, 
#             scene_data=scene_data,
#             weights=position_params['weights'],  # Use weights from the dictionary
#             sparsity_parameter=position_params['sparsity_parameter'],  # Use sparsity_parameter from the dictionary
#             # print_para=True,
#             print_prog=False
#         )
# print(found_UAV_positions)
# # print(hierachical_clustering_GU_records)
# # print(hierachical_clustering_GU_records[0])

# get_gu_to_uav_connections(ground_users, UAV_nodes, UAVInfo, blocks)
# # print_node(ground_users)
# # print(ground_users[0].connected_nodes)

# # visualize_hierarchical_clustering(ground_users, hierachical_clustering_GU_records[0], blocks ,scene)
# # visualize_hierarchical_clustering(ground_users, hierachical_clustering_GU_records, blocks ,scene)

# visualize_gu_by_connection(ground_users, blocks, scene)

# scene_visualization(ground_users, UAV_nodes, BS_nodes, scene_data, 0.3)


# print(gu_capacities_records)
# print(uav_load_records)

# visualize_capacity_and_load(gu_capacities_records, uav_load_records, True)
# visualize_capacity_and_load(gu_capacities_records, uav_load_records, False)

# try to store variables during position finding:
# import os
# file_path = r"C:\Users\logna\OneDrive\桌面\year 7 A\NICE LAB\Personal Meeting\2024-11-15\solving HC visualization problems\variables_output.txt"

# os.makedirs(os.path.dirname(file_path), exist_ok=True)

# with open(file_path, "w", encoding="utf-8") as file:
#     file.write("heatmap_for_visualization:\n")
#     file.write(str(heatmap_for_visualization) + "\n\n")
    
#     file.write("ground_users:\n")
#     file.write(str(ground_users) + "\n\n")
    
#     file.write("gu_capacities_records:\n")
#     file.write(str(gu_capacities_records) + "\n\n")
    
#     file.write("uav_load_records:\n")
#     file.write(str(uav_load_records) + "\n\n")

#visualize topology finding
# in each 
# print_node(UAV_nodes)
# best_state, max_reward, best_RS, reward_track, RS_track, best_reward_track, best_RS_track, best_backhaul_connection= find_best_backhaul_topology(
#     ground_users, 
#     UAV_nodes, 
#     BS_nodes, 
#     epsilon, 
#     episodes=training_episodes, 
#     scene_info = scene_data, 
#     reward_hyper=reward_hyper,
#     print_prog=True
# )

# print(reward_track)
# print(RS_track)
# print(best_reward_track)
# print(best_RS_track)

# visualize_scores(reward_track, RS_track, best_reward_track, best_RS_track)
# visualize_best_scores(best_reward_track, best_RS_track)

# ----
from node_functions import add_gu_to_simulation, add_or_remove_gu, set_baseline_backhaul_for_mid_scene

best_backhaul_connection = None

# create a baseline backhaul for comparison
baseline_UAV_nodes = generate_nodes(num_UAV, 1)
baseline_BS_nodes = generate_nodes(num_BS, 2)

baseline_BS_connections = [
    [0,1,4],
    [0,1,2,4]
]

# for mid scene:
baseline_UAV_positions = [(185, 100, 50), (126, 180, 50), (36, 147, 50), (36, 52, 50), (126, 19, 50)]

baseline_UAV_connections = [
    [2,4],
    [2,4],
    [0,1,3,4],
    [2],
    [0,1,2]
]
set_baseline_backhaul_for_mid_scene(baseline_UAV_nodes, baseline_UAV_positions, baseline_UAV_connections, baseline_BS_nodes, baseStation, baseline_BS_connections)
# print_node(baseline_UAV_nodes)

from connectivity_finding import get_backhaul_connection
# baseline_state = "010111 010111 111101 001000 111011"
# baseline_state = "010111 10111 1101 000 11 0"
baseline_state = "010111101111101000110"
baseline_backhaul_connection = get_backhaul_connection(baseline_state, baseline_UAV_nodes, baseline_BS_nodes, scene_data)

# print(baseline_backhaul_connection)
print_node(baseline_UAV_nodes)

get_gu_to_uav_connections(ground_users, baseline_UAV_nodes, UAVInfo, blocks)
scene_visualization(ground_users, baseline_UAV_nodes, baseline_BS_nodes, scene_data, 0.3)

for cur_time_frame in range(sim_time):  
    # add_or_remove_GU(ground_users, blocks, scene['xLength'], scene['yLength'], max_movement_distance, 2, add=True, print_info=True)
    # add_gu_to_simulation(ground_users,1)
    add_or_remove_gu(ground_users)
    
    gu_to_uav_connections, gu_to_bs_capacity = move_gu_and_update_connections(ground_users, blocks, scene['xLength'], scene['yLength'], max_movement_distance, UAV_nodes, UAVInfo, best_backhaul_connection)

    max_uav_load_number = max_count = max([item for sublist in gu_to_uav_connections.values() for item in sublist].count(x) for x in set([item for sublist in gu_to_uav_connections.values() for item in sublist]))

    if max_uav_load_number >= num_BS * float(constraint_hyper['max_uav_load_rate']) or min(gu_to_bs_capacity) < float(constraint_hyper['GUConstraint']):
        print("Finding positions")
        
        found_UAV_positions, hierachical_clustering_GU_records, gu_capacities_records, uav_load_records= find_optimal_uav_positions(
            ground_users=ground_users, 
            uavs=UAV_nodes, 
            scene_data=scene_data,
            weights=position_params['weights'],  # Use weights from the dictionary
            sparsity_parameter=position_params['sparsity_parameter'],  # Use sparsity_parameter from the dictionary
            # print_para=True,
            print_prog=False
        )

        print("Positions are found, finding connections")
        
        best_state, max_reward, best_RS, reward_track, RS_track, best_reward_track, best_RS_track, best_backhaul_connection= find_best_backhaul_topology(
            ground_users, 
            UAV_nodes, 
            BS_nodes, 
            epsilon, 
            episodes=training_episodes, 
            scene_info = scene_data, 
            reward_hyper=reward_hyper,
            print_prog=False
        )        
        print("Connections details are found, evaluating topo")
    else:
        print("Current topology is good enough, no topology refreshed is needed")
        max_reward, best_RS, current_backhaul_connection = reward(best_state, scene_data, ground_users, UAV_nodes, BS_nodes, reward_hyper)
    
    
    gu_to_uav_connections, gu_to_bs_capacity = get_gu_to_uav_connections(ground_users, UAV_nodes, UAVInfo, blocks, best_backhaul_connection)
    
    # scene_visualization(ground_users, UAV_nodes, BS_nodes, scene_data, 0.3)
    
    reward_TD.append(max_reward)
    RS_TD.append(best_RS)
    uav_connections_TD.append(gu_to_uav_connections)
    gu_capacity_TD.append(gu_to_bs_capacity)   

    # OL_TD.append(OverloadScore)
    # gu_capacity_TD.append(gu_to_bs_capacity)
    # UAV_capacity_TD.append(uav_to_bs_capacity)
    # UAV_overload_TD.append(uav_overload)

from visualization_functions import visualize_simulation, visualize_simulation_together

if sim_time > 0:

    print(reward_TD)
    print(RS_TD)
    print(gu_capacity_TD)
    print(uav_connections_TD)
    # print(min_gu_capacity_TD)
    # print(OL_TD)    

    # visualize_all_min_gu_capacity(min_gu_capacity_TD)
    # visualize_all_gu_capacity(gu_capacity_TD)
    # visualize_uav_capacity(all_uav_capacity=UAV_capacity_TD)
    # visualize_all_UAV_overload(all_UAV_overload=UAV_overload_TD)

    # print(uav_connections_TD)
    visualize_simulation(uav_connections_TD, gu_capacity_TD, num_UAV)
    visualize_simulation_together(uav_connections_TD, gu_capacity_TD, num_UAV)
    visualize_metrics(reward_TD, RS_TD)

    # print(gu_capacity_TD)
    # print(UAV_capacity_TD)
    # print(UAV_overload_TD)
    
        