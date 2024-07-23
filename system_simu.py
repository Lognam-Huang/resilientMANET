import json
from functions.print_nodes import print_nodes

from gu_movement import move_ground_users, simulate_and_visualize_movements

# Load scene data from JSON file
# with open('scene_data_system_overview.json', 'r') as file:
with open('scene_data_simple.json', 'r') as file:
# with open('scene_data.json', 'r') as file:
# with open('scene_data_mid.json', 'r') as file:
    scene_data = json.load(file)

blocks = scene_data['blocks']
scene = scene_data['scenario']
UAVInfo = scene_data['UAV']
baseStation = scene_data['baseStation']

num_GU = 5
num_UAV = 3
num_BS = len(scene_data['baseStation'])

from functions.generate_users import generate_users, add_or_remove_GU
ground_users = generate_users(num_GU, blocks, scene['xLength'], scene['yLength'])


from functions.generate_UAVs import generate_UAVs
defaultHeightUAV = 200
UAV_nodes = generate_UAVs(num_UAV, blocks, scene['xLength'], scene['yLength'], defaultHeightUAV, 10, 'basic UAV')

defaultHeightABS = 500
ABS_nodes = generate_UAVs(num_BS, blocks, scene['xLength'], scene['yLength'], defaultHeightABS, 10, 'Air Base Station')
for i in range(num_BS):
    ABS_nodes[i].set_position((baseStation[i]['bottomCorner'][0], baseStation[i]['bottomCorner'][1], baseStation[i]['height'][0]))

# Define parameters for 3D heatmap generation and UAV positioning
min_height = UAVInfo['min_height']
max_height = UAVInfo['max_height']

eps = 15
min_samples = 5

from functions.scene_visualization import scene_visualization
scene_visualization(ground_users=ground_users, UAV_nodes=UAV_nodes, air_base_station=ABS_nodes, blocks=blocks, scene_info=scene, line_alpha=0.5, show_axes_labels=True)

# Lognam: find UAV positions
from key_functions.uav_coverage_optimization import *
# plot_gu_capacities(max_capacities_tracks)
# plot_combined_gu_capacity_and_uav_load(max_capacities_tracks)

# Lognam: find UAV connection
from find_topo import *
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

ABS_coords = np.array(get_nodes_position(ABS_nodes))

# Q-learning hyperparameters
epsilon = 0.1
training_episodes=20

# Lognam: try to have TD
sim_time = 3

# Lognam: try to switch scenes
max_movement_distance = 20

reward_TD = []
RS_TD = []
OL_TD = []
gu_capacity_TD = []
UAV_capacity_TD = []
UAV_overload_TD = []


from classes.UAVMap import *

from simu_functions import *

constraint_hyper = {
    'rewardConstraint': 0.9,
    'RSConstraint': 0.8,
    'OLConstraint': 0.9
}

max_capacities_tracks = find_optimal_uav_positions(
    ground_users=ground_users, uavs=UAV_nodes, clustering_epsilon=eps, min_cluster_size=min_samples, obstacles=blocks, area_info=scene, min_altitude=min_height, max_altitude=max_height, uav_info=UAVInfo
    # , print_para=True
    , print_prog=True
)

best_state, max_reward, best_RS, best_OL, reward_track, RS_track, OL_track, cur_UAVMap = find_best_topology(ground_users, UAV_nodes, ABS_coords, epsilon, episodes=training_episodes, visualize=False, scene_info = scene_data, reward_hyper=reward_hyper
#   ,print_prog=True
)

uav_to_bs_connections = find_best_paths_to_bs(cur_UAVMap)
gu_to_uav_connections = get_gu_to_uav_connections(ground_users, UAV_nodes, UAVInfo, blocks)

rewardScore, ResilienceScore, OverloadScore, gu_to_bs_capacity, uav_to_bs_capacity, uav_overload = calculate_current_topology_metrics(
    ground_users, gu_to_uav_connections, uav_to_bs_connections, UAVInfo, cur_UAVMap, UAV_nodes, reward_hyper, scene_data, print_metrics=True
)

scene_visualization(ground_users=ground_users, UAV_nodes=UAV_nodes, air_base_station=ABS_nodes, blocks=blocks, scene_info=scene, connection_GU_UAV=gu_to_uav_connections, connection_UAV_BS=uav_to_bs_connections, line_alpha=0.5, show_axes_labels=False)


reward_TD.append(rewardScore)
RS_TD.append(ResilienceScore)
OL_TD.append(OverloadScore)
gu_capacity_TD.append(gu_to_bs_capacity)
UAV_capacity_TD.append(uav_to_bs_capacity)
UAV_overload_TD.append(uav_overload)



for cur_time_frame in range(sim_time):
    add_or_remove_GU(ground_users, blocks, scene['xLength'], scene['yLength'], max_movement_distance, 2, add=True, print_info=True)
    gu_to_uav_connections = move_gu_and_update_connections(ground_users, blocks, scene['xLength'], scene['yLength'], max_movement_distance, UAV_nodes, UAVInfo)

    rewardScore, ResilienceScore, OverloadScore, gu_to_bs_capacity, uav_to_bs_capacity, uav_overload = calculate_current_topology_metrics(
        ground_users, gu_to_uav_connections, uav_to_bs_connections, UAVInfo, cur_UAVMap, UAV_nodes, reward_hyper, scene_data, print_metrics=True
    )

    if rewardScore < constraint_hyper['rewardConstraint'] or OverloadScore < constraint_hyper['OLConstraint']:
        
        max_capacities_tracks = find_optimal_uav_positions(
            ground_users=ground_users, uavs=UAV_nodes, clustering_epsilon=eps, min_cluster_size=min_samples, obstacles=blocks, area_info=scene, min_altitude=min_height, max_altitude=max_height, uav_info=UAVInfo
            # , print_para=True
            , print_prog=True
        )

        best_state, max_reward, best_RS, best_OL, reward_track, RS_track, OL_track, cur_UAVMap = find_best_topology(ground_users, UAV_nodes, ABS_coords, epsilon, episodes=training_episodes, visualize=False, scene_info = scene_data, reward_hyper=reward_hyper
          ,print_prog=True
        )
        
        rewardScore, ResilienceScore, OverloadScore, gu_to_bs_capacity, uav_to_bs_capacity, uav_overload = calculate_current_topology_metrics(
            ground_users, gu_to_uav_connections, uav_to_bs_connections, UAVInfo, cur_UAVMap, UAV_nodes, reward_hyper, scene_data, print_metrics=True
        )

        gu_to_uav_connections = get_gu_to_uav_connections(ground_users, UAV_nodes, UAVInfo, blocks)
        uav_to_bs_connections = find_best_paths_to_bs(cur_UAVMap)
    
    scene_visualization(ground_users=ground_users, UAV_nodes=UAV_nodes, air_base_station=ABS_nodes, blocks=blocks, scene_info=scene, connection_GU_UAV=gu_to_uav_connections, connection_UAV_BS=uav_to_bs_connections, line_alpha=0.5, show_axes_labels=False)

    reward_TD.append(rewardScore)
    RS_TD.append(ResilienceScore)
    OL_TD.append(OverloadScore)
    gu_capacity_TD.append(gu_to_bs_capacity)
    UAV_capacity_TD.append(uav_to_bs_capacity)
    UAV_overload_TD.append(uav_overload)

    
        
if sim_time > 0:

    visualize_all_gu_capacity(all_gu_capacity=gu_capacity_TD)
    visualize_uav_capacity(all_uav_capacity=UAV_capacity_TD)
    visualize_all_UAV_overload(all_UAV_overload=UAV_overload_TD)

    visualize_metrics(reward_TD, RS_TD, OL_TD)

    print(gu_capacity_TD)
    print(UAV_capacity_TD)
    print(UAV_overload_TD)

    print(reward_TD)
    print(RS_TD)
    print(OL_TD)    