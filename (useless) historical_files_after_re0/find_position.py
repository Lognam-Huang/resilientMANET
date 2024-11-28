import json

from functions.get_3D_heatmap import get_max_cluster_point
from key_functions.uav_coverage_optimization import *
from functions.generate_users import generate_users
from functions.generate_UAVs import generate_UAVs
from functions.scene_visualization import scene_visualization
from functions.print_nodes import print_nodes, get_nodes_position

# Load scene data from JSON file
with open('scene_data_simple.json', 'r') as file:
    scene_data = json.load(file)

blocks = scene_data['blocks']
UAVInfo = scene_data['UAV']
scene = scene_data['scenario']

# Generate ground users and UAVs
ground_users = generate_users(15, blocks, scene['xLength'], scene['yLength'])
UAVs = generate_UAVs(3, blocks, scene['xLength'], scene['yLength'], 50, 3, "basic UAV")

# Set specific positions for some ground users
specified_positions = [(3, 3, 0), (24, 24, 0), (3, 8, 0), (8, 2, 0), (24, 22, 0), (21, 24, 0)]
for gu, pos in zip(ground_users, specified_positions):
    gu.set_position(pos)

# Define parameters for 3D heatmap generation and UAV positioning
min_height = 10
max_height = 15
eps = 15
min_samples = 5

# Find optimal points for UAVs to cover ground users
# max_capacities_tracks = get_max_cluster_point(
#     ground_users, UAVs, eps, min_samples, blocks, scene, min_height, max_height, UAVInfo
# )

max_capacities_tracks = find_optimal_uav_positions(
    ground_users, UAVs, eps, min_samples, blocks, scene, min_height, max_height, UAVInfo
)

# print("All uavs nodes are at:")
# print_nodes(ground_users)
# print_nodes(UAVs, True)

# get positions for MAPPO
UAV_positions = get_nodes_position(UAVs)
print(UAV_positions)

# Plot the capacities of ground users over time
plot_gu_capacities(max_capacities_tracks)

# this is a failure
# plot_gu_uav_statistics(max_capacities_tracks)

# plot 2 images separatedly
# plot_gu_summary_and_uav_load(max_capacities_tracks)

# plot 2 images together
plot_combined_gu_capacity_and_uav_load(max_capacities_tracks)

# Visualize the scene with ground users, UAVs, and blocks
# scene_visualization(ground_users, UAVs, blocks=blocks, scene_info=scene)
