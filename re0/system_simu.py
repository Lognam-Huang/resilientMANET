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
    'sparsity_parameter': 10  # Controls the density of the heatmap
    ,
    "eps" : 8,
    "min_samples" : 2
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
training_episodes= 50

# Lognam: set simulation time
sim_time = 15

# Lognam: try to switch scenes
max_movement_distance = 150

constraint_hyper = {
    'rewardConstraint': 0.8,
    'GUConstraint': 100000000
}

rewardScore = 0
# OverloadScore = 0

reward_TD = []
RS_TD = []
OL_TD = []
gu_capacity_TD = []
UAV_capacity_TD = []
UAV_overload_TD = []

from node_functions import move_ground_users, get_gu_to_uav_connections, move_gu_and_update_connections

print(move_gu_and_update_connections(ground_users, blocks, scene['xLength'], scene['yLength'], max_movement_distance, UAV_nodes, UAVInfo))


from position_finding import find_optimal_uav_positions
from connectivity_finding import find_best_backhaul_topology, reward, set_connected_edges
from visualization_functions import scene_visualization

# print_node(UAV_nodes, -1, True)

for cur_time_frame in range(sim_time):  
    # add_or_remove_GU(ground_users, blocks, scene['xLength'], scene['yLength'], max_movement_distance, 2, add=True, print_info=True)
    if cur_time_frame > 0:
        gu_to_uav_connections = move_gu_and_update_connections(ground_users, blocks, scene['xLength'], scene['yLength'], max_movement_distance, UAV_nodes, UAVInfo)

        rewardScore, ResilienceScore = reward(best_state, scene_data, ground_users, UAV_nodes, BS_nodes, reward_hyper)

    # if rewardScore < constraint_hyper['rewardConstraint']:
    if True:

        print("Finding positions")
        
        max_capacities_tracks = find_optimal_uav_positions(
            ground_users=ground_users, 
            uavs=UAV_nodes, 
            scene_data=scene_data,
            weights=position_params['weights'],  # Use weights from the dictionary
            sparsity_parameter=position_params['sparsity_parameter'],  # Use sparsity_parameter from the dictionary
            # print_para=True,
            print_prog=False
        )

        print("Positions are found, finding connections")
        
        best_state, max_reward, best_RS, reward_track, RS_track= find_best_backhaul_topology(
            ground_users, 
            UAV_nodes, 
            BS_nodes, 
            epsilon, 
            episodes=training_episodes, 
            scene_info = scene_data, 
            reward_hyper=reward_hyper,
            print_prog=True
        )

        # print(best_state)
        
        print("Connections details are found, evaluating topo")
        get_gu_to_uav_connections(ground_users, UAV_nodes, UAVInfo, blocks)

        rewardScore, ResilienceScore = reward(best_state, scene_data, ground_users, UAV_nodes, BS_nodes, reward_hyper)

    else:
        print("Current topology is good enough, no topology refreshed is needed")
    
    # print_node(UAV_nodes, -1, True)

    set_connected_edges(best_state, UAV_nodes, BS_nodes)
    
    scene_visualization(ground_users, UAV_nodes, BS_nodes, scene_data, 0.3)

    reward_TD.append(rewardScore)
    RS_TD.append(ResilienceScore)
    # OL_TD.append(OverloadScore)
    # gu_capacity_TD.append(gu_to_bs_capacity)
    # UAV_capacity_TD.append(uav_to_bs_capacity)
    # UAV_overload_TD.append(uav_overload)

    
        