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
num_BS = 1

from functions.generate_users import generate_users
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
max_capacities_tracks = find_optimal_uav_positions(
    ground_users=ground_users, uavs=UAV_nodes, clustering_epsilon=eps, min_cluster_size=min_samples, obstacles=blocks, area_info=scene, min_altitude=min_height, max_altitude=max_height, uav_info=UAVInfo
    # , print_para=True
    , print_prog=True
)

print(max_capacities_tracks)

plot_gu_capacities(max_capacities_tracks)
plot_combined_gu_capacity_and_uav_load(max_capacities_tracks)

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

UAV_coords = np.array(get_nodes_position(UAV_nodes))

# Q-learning hyperparameters
epsilon = 0.1
training_episodes=20

scene_visualization(ground_users=ground_users, UAV_nodes=UAV_nodes, air_base_station=ABS_nodes, blocks=blocks, scene_info=scene, line_alpha=0.5, show_axes_labels=False)

best_state, max_reward, reward_track, RS_track, OL_track, cur_UAVMap = find_best_topology(UAV_coords, ABS_coords, epsilon, episodes=training_episodes, visualize=False, scene_info = scene_data, reward_hyper=reward_hyper
                                                                                          ,print_prog=True
                                                                                          )

# Lognam: try to have TD
sim_time = 0

# Lognam: try to switch scenes
max_movement_distance = 20

max_reward_TD = []
reward_track_TD = []

max_RS_TD = []
max_OL_TD = []


# print(max_reward)

# print(RS_track)
# print(max(RS_track))

max_reward_TD.append(max_reward)
max_RS_TD.append(max(RS_track))
max_OL_TD.append(max(OL_track))

from classes.UAVMap import *
print(cur_UAVMap)

uav_to_bs_connections = find_best_paths_to_bs(cur_UAVMap)
gu_to_uav_connections = extract_gu_to_uav_connections(ground_users)

scene_visualization(ground_users=ground_users, UAV_nodes=UAV_nodes, air_base_station=ABS_nodes, blocks=blocks, scene_info=scene, connection_GU_UAV=gu_to_uav_connections, connection_UAV_BS=uav_to_bs_connections, line_alpha=0.5, show_axes_labels=False)

# print(uav_to_bs_connections)
# print(gu_to_uav_connections)

from simu_functions import *

GU_capacity, UAV_capacity, UAV_overload = calculate_capacity_and_overload(ground_users=ground_users, gu_to_uav_connections=gu_to_uav_connections, uav_to_bs_connections=uav_to_bs_connections, uav_info=UAVInfo, cur_UAVMap=cur_UAVMap, UAV_nodes=UAV_nodes)

# print(GU_capacity)
# print(UAV_capacity)
# print(UAV_overload)


all_gu_capacity = [] 
all_gu_capacity.append(GU_capacity)

all_UAV_capacity = []
all_UAV_capacity.append(UAV_capacity)
all_UAV_overload = []
all_UAV_overload.append(UAV_overload)

# print(cur_UAVMap.allPaths.get(0, []))

for cur_time_frame in range(sim_time):
    move_ground_users(ground_users, blocks, scene['xLength'], scene['yLength'], max_movement_distance)

    # print("Moving GUs")

    max_capacities_tracks = find_optimal_uav_positions(
        ground_users=ground_users, uavs=UAV_nodes, clustering_epsilon=eps, min_cluster_size=min_samples, obstacles=blocks, area_info=scene, min_altitude=min_height, max_altitude=max_height, uav_info=UAVInfo
        # , print_para=True
        , print_prog=True
    )
    
    # print("Finding positions")

    UAV_coords = np.array(get_nodes_position(UAV_nodes))
    best_state, max_reward, reward_track, RS_track, OL_track, cur_UAVMap = find_best_topology(UAV_coords, ABS_coords, epsilon, episodes=50, visualize=False, scene_info = scene_data, reward_hyper=reward_hyper
        , print_prog=True
        )

    max_reward_TD.append(max_reward)
    max_RS_TD.append(max(RS_track))
    max_OL_TD.append(max(OL_track))

    print("Start connecting")

    print(best_state)
    print(max_reward)

    uav_to_bs_connections = find_best_paths_to_bs(cur_UAVMap)
    gu_to_uav_connections = extract_gu_to_uav_connections(ground_users)

    GU_capacity, UAV_capacity, UAV_overload = calculate_capacity_and_overload(ground_users=ground_users, gu_to_uav_connections=gu_to_uav_connections, uav_to_bs_connections=uav_to_bs_connections, uav_info=UAVInfo, cur_UAVMap=cur_UAVMap, UAV_nodes=UAV_nodes)
    all_gu_capacity.append(GU_capacity)
    all_UAV_capacity.append(UAV_capacity)
    all_UAV_overload.append(UAV_overload)

    # scene_visualization(ground_users=ground_users, UAV_nodes=UAV_nodes, air_base_station=ABS_nodes, blocks=blocks, scene_info=scene, connection_GU_UAV=gu_to_uav_connections, connection_UAV_BS=uav_to_bs_connections, line_alpha=0.5, show_axes_labels=False)




if sim_time > 0:

    visualize_all_gu_capacity(all_gu_capacity=all_gu_capacity)
    visualize_uav_capacity(all_uav_capacity=all_UAV_capacity)
    visualize_all_UAV_overload(all_UAV_overload=all_UAV_overload)

    visualize_metrics(max_reward_TD, max_RS_TD, max_OL_TD)

    print(all_gu_capacity)
    print(all_UAV_capacity)
    print(all_UAV_overload)

    print(max_reward_TD)
    print(max_RS_TD)
    print(max_OL_TD)    