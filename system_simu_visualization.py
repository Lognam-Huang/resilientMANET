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
BS_nodes = generate_nodes(num_BS, 2)

for i in range(num_BS):
    BS_nodes[i].set_position((baseStation[i]['bottomCorner'][0], baseStation[i]['bottomCorner'][1], baseStation[i]['height'][0]))

position_params = {
    'weights': {
        'GU': 8,  # Weight for ground user connections
        'UAV': 2,  # Weight for UAV-to-UAV connections
        'BS': 1   # Weight for base station connections
    },
    'sparsity_parameter': 5  # Controls the density of the heatmap
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
# training_episodes= 200
training_episodes= 20

# Lognam: set simulation time
sim_time = 10

# Lognam: try to switch scenes
max_movement_distance = 200

constraint_hyper = {
    'rewardConstraint': 0.8,
    'GUConstraint': 100000000,
    'max_uav_load_rate': 0.5
}


reward_TD = []
RS_TD = []
OL_TD = []
gu_capacity_TD = []
UAV_capacity_TD = []
UAV_overload_TD = []

uav_connections_TD = []

min_gu_capacity_TD = []

from node_functions import get_gu_to_uav_connections, move_gu_and_update_connections

for i in range(10):
    move_gu_and_update_connections(ground_users, blocks, scene['xLength'], scene['yLength'], max_movement_distance, UAV_nodes, UAVInfo, None)

from position_finding import find_optimal_uav_positions, generate_3D_heatmap
from connectivity_finding import find_best_backhaul_topology, reward, set_connected_edges
from visualization_functions import scene_visualization, visualize_all_gu_capacity, visualize_metrics, visualize_all_min_gu_capacity, visualize_heatmap_slices, visualize_hierarchical_clustering, visualize_capacity_and_load, visualize_scores, visualize_best_scores, visualize_gu_by_connection


# ----
# try to visualize for paper, section VII-B, C, D

# heatmap_for_visualization, best_position_for_visualization, max_connection_score_for_visualization, min_gu_bottleneck_for_visualization = generate_3D_heatmap(ground_users, scene_data, position_params['weights'], position_params['sparsity_parameter'])
# # print(heatmap_for_visualization)

# selected_heights = [10, 15]
# # selected_heights = [50, 55]
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

# get_gu_to_uav_connections(ground_users, UAV_nodes, UAVInfo, blocks)
# visualize_gu_by_connection(ground_users, blocks, scene)

# scene_visualization(ground_users, UAV_nodes, BS_nodes, scene_data, 0.3)


# print(gu_capacities_records)
# print(uav_load_records)

# visualize_capacity_and_load(gu_capacities_records, uav_load_records, True)
# visualize_capacity_and_load(gu_capacities_records, uav_load_records, False)

# # visualize topology finding
# # print_node(UAV_nodes)
# best_state, max_reward, best_RS, reward_track, RS_track, best_reward_track, best_RS_track, best_backhaul_connection= find_best_backhaul_topology(
#     ground_users, 
#     UAV_nodes, 
#     BS_nodes, 
#     epsilon, 
#     episodes=training_episodes, 
#     scene_info = scene_data, 
#     reward_hyper=reward_hyper,
#     print_prog=True,
#     initialize_as_all_0=True
# )

# print(reward_track)
# print(RS_track)
# print(best_reward_track)
# print(best_RS_track)

# scene_visualization(ground_users, UAV_nodes, BS_nodes, scene_data, 0.3)
# visualize_scores(reward_track, RS_track, best_reward_track, best_RS_track)
# visualize_best_scores(best_reward_track, best_RS_track)

# # redraw figure 4 for ACM paper
gu_fixed_position = [(20,8,0), (61,5,0), (80,1,0),
                     (70,30,0), (70,115,0), (52, 120,0),
                     (155,35,0), (195,50,0), 
                     (180,128,0), (183,132,0),(190,115,0),
                     (185,160,0)
                     ]


for i in range(len(gu_fixed_position)):
    ground_users[i].set_position(gu_fixed_position[i])


# uav_fixed_position = [(35,15,50),
#                       (90,50,50),
#                       (175,50,50),
#                       (190,135,50),
#                       (190,155,50)
#                      ]

# try to visualize DRL
uav_fixed_position = [(0,5,50),
                      (3,1,50),
                      (3,13,50),
                      (5,8,50),
                      (6,2,50)
                     ]


for i in range(len(uav_fixed_position)):
    UAV_nodes[i].set_position(uav_fixed_position[i])

# # state that is with backhaul connection
state_for_mid_scene = "000010111111110111110"
# # state that is without backhaul connection
# state_for_mid_scene = "000000000000000000000"


from connectivity_finding import get_backhaul_connection
backhaul_connection = get_backhaul_connection(state_for_mid_scene, UAV_nodes, BS_nodes, scene_data)
get_gu_to_uav_connections(ground_users, UAV_nodes, UAVInfo, blocks, backhaul_connection)

scene_visualization(ground_users, UAV_nodes, BS_nodes, scene_data, 0.2)

from visualization_functions import visualize_hierarchical_clustering_result_in_scene
visualize_gu_by_connection(ground_users, blocks, scene, BS_nodes)
visualize_hierarchical_clustering_result_in_scene(ground_users, UAV_nodes, blocks, scene, BS_nodes, 0.2)

# redraw figure 5

# gu_capacities_records = [
#     {0: 0, 1: 0, 2: 0}, 
#     {0: 180000000, 1: 950000000, 2: 9017824902}, 
#     {0: 300000000, 1: 1050000000, 2: 10195053574.893757},
#     {0: 6000000000, 1: 10500000000, 2: 10500000000, 3: 10500000000, 4:10500000000},
#     {0: 6800000000, 1: 11500000000, 2: 10500000000, 3: 10500000000, 4:10500000000},
#     {0: 6800000000, 1: 11500000000, 2: 11500000000, 3: 10500000000, 4:10500000000}
#     ]
# uav_load_records = [{0: 12}, {0: 8, 1: 4}, {0: 7, 1: 3, 2: 2},{0: 7, 1: 3, 2: 1, 3: 1},{0: 5, 1: 3, 2: 1, 3:3 },{0: 3, 1: 3, 2: 2, 3: 3, 4: 1}]
# visualize_capacity_and_load(gu_capacities_records, uav_load_records, False)

# redraw figure 7
from visualization_functions import visualize_simulation_with_multiple_baselines, visualize_simulation_with_multiple_baselines_styled_2

import pandas as pd
import ast
# read_result_test= pd.read_csv("experiment_result_mid.csv")

# read_uav_connections_TD = read_result_test["UAV Connections"].apply(ast.literal_eval).tolist()
# read_uav_gu_capacity_TD = read_result_test["GU Capacity"].apply(ast.literal_eval).tolist()


# read_baseline_1_result= pd.read_csv("experiment_result_mid_baseline_1_stable.csv")

# read_baseline_1_uav_connections_TD = read_baseline_1_result["UAV Connections"].apply(ast.literal_eval).tolist()
# read_baseline_1_uav_gu_capacity_TD = read_baseline_1_result["GU Capacity"].apply(ast.literal_eval).tolist()

# read_baseline_2_result= pd.read_csv("experiment_result_mid_baseline_2_stable.csv")

# read_baseline_2_uav_connections_TD = read_baseline_2_result["UAV Connections"].apply(ast.literal_eval).tolist()
# read_baseline_2_uav_gu_capacity_TD = read_baseline_2_result["GU Capacity"].apply(ast.literal_eval).tolist()

# visualize_simulation_with_multiple_baselines_styled_2(read_uav_connections_TD, read_uav_gu_capacity_TD, read_baseline_1_uav_connections_TD, read_baseline_1_uav_gu_capacity_TD, read_baseline_2_uav_connections_TD, read_baseline_2_uav_gu_capacity_TD, 5, 3)

# redraw figure 8
# read_result_test= pd.read_csv("experiment_result_mid_dynamic.csv")

# read_uav_connections_TD = read_result_test["UAV Connections"].apply(ast.literal_eval).tolist()
# read_uav_gu_capacity_TD = read_result_test["GU Capacity"].apply(ast.literal_eval).tolist()

# read_baseline_1_result= pd.read_csv("experiment_result_mid_baseline_1_dynamic.csv")

# read_baseline_1_uav_connections_TD = read_baseline_1_result["UAV Connections"].apply(ast.literal_eval).tolist()
# read_baseline_1_uav_gu_capacity_TD = read_baseline_1_result["GU Capacity"].apply(ast.literal_eval).tolist()

# read_baseline_2_result= pd.read_csv("experiment_result_mid_baseline_2_dynamic.csv")

# read_baseline_2_uav_connections_TD = read_baseline_2_result["UAV Connections"].apply(ast.literal_eval).tolist()
# read_baseline_2_uav_gu_capacity_TD = read_baseline_2_result["GU Capacity"].apply(ast.literal_eval).tolist()

# visualize_simulation_with_multiple_baselines_styled_2(read_uav_connections_TD, read_uav_gu_capacity_TD, read_baseline_1_uav_connections_TD, read_baseline_1_uav_gu_capacity_TD, read_baseline_2_uav_connections_TD, read_baseline_2_uav_gu_capacity_TD, 5, 3)