import json

# Load scene data from JSON file
with open('scene_data_hard.json', 'r') as file:
# with open('scene_data_simple.json', 'r') as file:
# with open('scene_data.json', 'r') as file:
# with open('scene_data_mid.json', 'r') as file:
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
    # 'sparsity_parameter': 1  # Controls the density of the heatmap

    # just for simple test, and now for quicker calculation for hard scene
    'sparsity_parameter': 50  # for hard scene simulation
    # 'sparsity_parameter': 10  # just for test
}


# Lognam: find UAV connection
reward_hyper = {
    'DRPenalty': 0.5,
    'BPHopConstraint': 4,
    #change this for hard scene simulation
    'BPDRConstraint': 10000000,

    'droppedRatio': 0.2,
    'ratioDR': 0.6,
    'ratioBP': 0.4,
    
    'weightDR': 0.3,
    'weightBP': 0.4,
    'weightNP': 0.3,

    # # hyperset 2
    # 'weightDR': 0.7,
    # 'weightBP': 0.15,
    # 'weightNP': 0.15,

    # # hyperset 3
    # 'weightDR': 0.2,
    # 'weightBP': 0.4,
    # 'weightNP': 0.4,

    'overloadConstraint': 10000
}

# Q-learning hyperparameters
q_hyper = {
    # just for simple test, now also for hard scene simulation
    'epsilon': 0.3,
    'training_episodes': 15

    # 'epsilon': 0.4,
    # 'training_episodes': 200
    # 'training_episodes': 400
}

# training_episodes= 200
# training_episodes= 20

# Lognam: set simulation time
sim_time = 30

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

# for storing information
GU_position_TD = []
UAV_position_TD = []
state_TD = []

from node_functions import get_gu_to_uav_connections, move_gu_and_update_connections, get_nodes_position
from position_finding import find_optimal_uav_positions
from connectivity_finding import find_best_backhaul_topology, reward
from visualization_functions import scene_visualization, visualize_metrics
from node_functions import add_or_remove_gu, set_baseline_backhaul, get_gu_info_and_update_connections

best_backhaul_connection = None

# def append_and_save_csv(reward, rs, uav_connections, gu_capacity, gu_positions, uav_positions, state, file_path):
#     new_data = {
#         "GU Position": [gu_positions],
#         "UAV Position": [uav_positions],
#         "GU Capacity": [gu_capacity],
#         "UAV Connections": [uav_connections],
        
#         "State": [state],
#         "Reward": [reward],
#         "RS": [rs],
        
#     }
#     pd.DataFrame(new_data).to_csv(file_path, mode='a', header=False, index=False)


import pandas as pd

# ground_users_positions_simple_stable = pd.read_csv("ground_user_positions_for_simple_scene_50_stable.csv")
# ground_users_positions_mid_stable = pd.read_csv("ground_user_positions_for_mid_scene_50_stable.csv")
# ground_users_positions_mid_dynamic = pd.read_csv("ground_user_positions_for_mid_scene_50_dynamic.csv")
ground_users_positions_hard_stable = pd.read_csv("ground_user_positions_for_hard_scene_50_stable.csv")


# for cur_time_frame in range(sim_time): 
# for hard scene simulation optimization
for cur_time_frame in range(0, sim_time, 6): 

    # this functino is used after we make user of pre-defined GU data
    # ground_users, gu_to_uav_connections, gu_to_bs_capacity = get_gu_info_and_update_connections(ground_users_positions_simple_stable, cur_time_frame, blocks, UAV_nodes, UAVInfo, best_backhaul_connection)
    # ground_users, gu_to_uav_connections, gu_to_bs_capacity = get_gu_info_and_update_connections(ground_users_positions_mid_stable, cur_time_frame, blocks, UAV_nodes, UAVInfo, best_backhaul_connection)
    # ground_users, gu_to_uav_connections, gu_to_bs_capacity = get_gu_info_and_update_connections(ground_users_positions_mid_dynamic, cur_time_frame, blocks, UAV_nodes, UAVInfo, best_backhaul_connection)
    
    ground_users, gu_to_uav_connections, gu_to_bs_capacity = get_gu_info_and_update_connections(ground_users_positions_hard_stable, cur_time_frame, blocks, UAV_nodes, UAVInfo, best_backhaul_connection)

    # print_node(ground_users, -1, True)
    max_uav_load_number = max_count = max([item for sublist in gu_to_uav_connections.values() for item in sublist].count(x) for x in set([item for sublist in gu_to_uav_connections.values() for item in sublist]))

    # scene_visualization(ground_users, UAV_nodes=None, air_base_station=BS_nodes, scene_info=scene_data, line_alpha=0.3)

    if max_uav_load_number >= num_BS * float(constraint_hyper['max_uav_load_rate']) or min(gu_to_bs_capacity) < float(constraint_hyper['GUConstraint']):
        print("Finding positions")
        
        found_UAV_positions, hierachical_clustering_GU_records, gu_capacities_records, uav_load_records= find_optimal_uav_positions(
            ground_users=ground_users, 
            uavs=UAV_nodes, 
            scene_data=scene_data,
            weights=position_params['weights'],  # Use weights from the dictionary
            sparsity_parameter=position_params['sparsity_parameter'],  # Use sparsity_parameter from the dictionary
            # print_para=True,
            # print_prog=False
            print_prog=True
        )



        print("Positions are found, finding connections")
        
        best_state, max_reward, best_RS, reward_track, RS_track, best_reward_track, best_RS_track, best_backhaul_connection= find_best_backhaul_topology(
            ground_users, 
            UAV_nodes, 
            BS_nodes, 
            q_hyper['epsilon'], 
            episodes=q_hyper['training_episodes'], 
            scene_info = scene_data, 
            reward_hyper=reward_hyper,
            print_prog=False
            # print_prog=True,
            initialize_as_all_0=False
        )        
        print("Connections details are found, evaluating topo")
    else:
        print("Current topology is good enough, no topology refreshed is needed")
        max_reward, best_RS, current_backhaul_connection = reward(best_state, scene_data, ground_users, UAV_nodes, BS_nodes, reward_hyper)
    
    gu_to_uav_connections, gu_to_bs_capacity = get_gu_to_uav_connections(ground_users, UAV_nodes, UAVInfo, blocks, best_backhaul_connection)

    # scene_visualization(ground_users, UAV_nodes, BS_nodes, scene_data, 0.3)
    
    # this is just for invisible-connection visualization
    # scene_visualization(ground_users, UAV_nodes, BS_nodes, scene_data, 0)

    
    reward_TD.append(max_reward)
    RS_TD.append(best_RS)
    uav_connections_TD.append(gu_to_uav_connections)
    gu_capacity_TD.append(gu_to_bs_capacity)   

    GU_position_TD.append(get_nodes_position(ground_users))
    UAV_position_TD.append(get_nodes_position(UAV_nodes))
    state_TD.append(best_state)

    # append_and_save_csv(
    #     max_reward,
    #     best_RS,
    #     gu_to_uav_connections,
    #     gu_to_bs_capacity,
    #     GU_position_TD,
    #     UAV_position_TD,
    #     best_state,
    #     "experiment_result_hard_stable.csv",
    # )

# record data
recorded_data = {
    "GU Position": [tuple(pos) for pos in GU_position_TD], 
    "UAV Position": [tuple(pos) for pos in UAV_position_TD],
    "State": state_TD,
    "UAV Connections": uav_connections_TD,
    "GU Capacity": gu_capacity_TD,
}

recorded_df = pd.DataFrame(recorded_data)

# recorded_df.to_csv("experiment_result_test.csv", mode='a', header=False, index=False)

# recorded_df.to_csv("experiment_result_mid_hyper3.csv", index=False)
# recorded_df.to_csv("experiment_result_mid_dynamic.csv", index=False)

# recorded_df.to_csv("experiment_result_hard_stable_2.csv", mode='a', header=False, index=False)
recorded_df.to_csv("experiment_result_hard_stable_3.csv", mode='a', header=False, index=False)

recorded_hypers = {
    "reward_track": reward_track, 
    "RS_track": RS_track,
    "best_reward_track": best_reward_track,
    "best_RS_track": best_RS_track,
}

recorded_hyper_df = pd.DataFrame(recorded_hypers)

# recorded_hyper_df.to_csv("experiment_result_scores_test.csv", mode='a', header=False, index=False)
# recorded_hyper_df.to_csv("experiment_result_hard_stable_scores_2.csv", mode='a', header=False, index=False)
recorded_hyper_df.to_csv("experiment_result_hard_stable_scores_3.csv", mode='a', header=False, index=False)

from visualization_functions import visualize_simulation, visualize_simulation_together, visualize_simulation_with_baseline, visualize_scores
if sim_time > 0:

    # print(uav_connections_TD)
    # print(gu_capacity_TD)
    # # print(baseline_uav_connections_TD)
    # # print(baseline_gu_capacity_TD)
    # print(GU_position_TD)

    visualize_simulation(uav_connections_TD, gu_capacity_TD, num_UAV)
    # visualize_simulation(baseline_uav_connections_TD, baseline_gu_capacity_TD, num_UAV)
    # visualize_simulation_with_baseline(uav_connections_TD, gu_capacity_TD, baseline_uav_connections_TD, baseline_gu_capacity_TD, num_UAV)
    # visualize_simulation_together(uav_connections_TD, gu_capacity_TD, num_UAV)
    # visualize_metrics(reward_TD, RS_TD)

    visualize_scores(reward_track, RS_track, best_reward_track, best_RS_track)

        