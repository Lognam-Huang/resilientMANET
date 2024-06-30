import json
from functions.print_nodes import print_nodes

from gu_movement import move_ground_users, simulate_and_visualize_movements

# Load scene data from JSON file
# with open('scene_data_system_overview.json', 'r') as file:
with open('scene_data_simple.json', 'r') as file:
    scene_data = json.load(file)

blocks = scene_data['blocks']
scene = scene_data['scenario']
UAVInfo = scene_data['UAV']

num_GU = 5
num_UAV = 3

from functions.generate_users import generate_users
ground_users = generate_users(num_GU, blocks, scene['xLength'], scene['yLength'])


from functions.generate_UAVs import generate_UAVs
defaultHeightUAV = 200
UAV_nodes = generate_UAVs(num_UAV, blocks, scene['xLength'], scene['yLength'], defaultHeightUAV, 10, 'basic UAV')


# Define parameters for 3D heatmap generation and UAV positioning
# min_height = 190
# max_height = 210
min_height = 10
max_height = 15
eps = 15
min_samples = 5

from functions.scene_visualization import scene_visualization
scene_visualization(ground_users=ground_users, UAV_nodes=UAV_nodes, blocks=blocks, scene_info=scene, line_alpha=0.5, show_axes_labels=False)

# print_nodes(ground_users, True)
# print_nodes(UAV_nodes, True)

# Lognam: find UAV positions
from key_functions.uav_coverage_optimization import *
max_capacities_tracks = find_optimal_uav_positions(
    ground_users=ground_users, uavs=UAV_nodes, clustering_epsilon=eps, min_cluster_size=min_samples, obstacles=blocks, area_info=scene, min_altitude=min_height, max_altitude=max_height, uav_info=UAVInfo
    # , print_para=True
)

# scene_visualization(ground_users=ground_users, UAV_nodes=UAV_nodes, blocks=blocks, scene_info=scene, line_alpha=0.5, show_axes_labels=False)

# print_nodes(ground_users, True)
# print_nodes(UAV_nodes, True)
# plot_gu_capacities(max_capacities_tracks)
# plot_combined_gu_capacity_and_uav_load(max_capacities_tracks)
# scene_visualization(ground_users=ground_users, UAV_nodes=UAV_nodes, blocks=blocks, scene_info=scene, line_alpha=0.5, show_axes_labels=False)


# defaultHeightABS = 500
# ABSNodes = generate_UAVs(1, blocks, scene['xLength'], scene['yLength'], defaultHeightABS, 10, 'Air Base Station')

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

ABS_coords = np.array([
    (20, 20, 18)
])

UAV_coords = np.array(get_nodes_position(UAV_nodes))

# Q-learning hyperparameters
epsilon = 0.1
best_state, max_reward, reward_track, RS_track, OL_track = find_best_topology(UAV_coords, ABS_coords, epsilon, episodes=50, visualize=False)

print(best_state)
print(max_reward)

# Lognam: try to have TD
sim_time = 20
max_movement_distance = 5

max_reward_TD = []
reward_track_TD = []

max_reward_TD.append(max_reward)

move_ground_users(ground_users, blocks, scene['xLength'], scene['yLength'], max_movement_distance)

print_nodes(ground_users, True)
print_nodes(UAV_nodes, True)
scene_visualization(ground_users=ground_users, UAV_nodes=UAV_nodes, blocks=blocks, scene_info=scene, line_alpha=0.5, show_axes_labels=True)

print_nodes(ground_users, True)
print_nodes(UAV_nodes, True)

print("Moving GUs")

max_capacities_tracks = find_optimal_uav_positions(
    ground_users=ground_users, uavs=UAV_nodes, clustering_epsilon=eps, min_cluster_size=min_samples, obstacles=blocks, area_info=scene, min_altitude=min_height, max_altitude=max_height, uav_info=UAVInfo
    # ,
    # , print_para=True
)

print("Finding positions")

UAV_coords = np.array(get_nodes_position(UAV_nodes))
best_state, max_reward, reward_track, RS_track, OL_track = find_best_topology(UAV_coords, ABS_coords, epsilon, episodes=50, visualize=False)


# for cur_time_frame in range(sim_time):
#     move_ground_users(ground_users, blocks, scene['xLength'], scene['yLength'], max_movement_distance)

#     print("Moving GUs")

#     max_capacities_tracks = find_optimal_uav_positions(
#         ground_users=ground_users, uavs=UAV_nodes, clustering_epsilon=eps, min_cluster_size=min_samples, obstacles=blocks, area_info=scene, min_altitude=min_height, max_altitude=max_height, uav_info=UAVInfo
#         # , print_para=True
#     )
    
#     print("Finding positions")

#     UAV_coords = np.array(get_nodes_position(UAV_nodes))
#     best_state, max_reward, reward_track, RS_track, OL_track = find_best_topology(UAV_coords, ABS_coords, epsilon, episodes=50, visualize=False)

#     # max_reward_TD.append(max_reward)
#     print("Start connecting")

#     print(best_state)
#     print(max_reward)

plt.plot(max_reward_TD, label='Reward TD', color='blue')