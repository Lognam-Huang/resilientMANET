import numpy as np
from sklearn.cluster import DBSCAN

# Assuming these utility functions are defined elsewhere in your project
# def path_is_blocked(obstacles, nodeA, nodeB):
#     # Function to determine if the path between nodeA and nodeB is blocked by any obstacles
#     pass

# def calculate_data_rate(UAVInfo, UAV_position, ground_user_position, block_or_not):
#     # Function to calculate data rate between a UAV and a ground user based on position and obstacles
#     pass

def is_position_inside_block(position, blocks):
    x, y, z = position
    for block in blocks:
        bx, by, bz = block["bottomCorner"]
        sx, sy = block["size"]
        h = block['height']
        if (bx <= x <= bx + sx) and (by <= y <= by + sy) and (bz <= z <= bz + h):
            return True
    return False

from functions.path_is_blocked import path_is_blocked
from classes.Nodes import Nodes
from sklearn.cluster import DBSCAN

from functions.calculate_data_rate import *

def generate_3D_heatmap(ground_users, obstacles, area_dimensions, min_altitude, max_altitude, weights, uav_info, sparsity_parameter=1, target_user_indices=None):
    """
    Generate a 3D heatmap of connection scores and GU bottlenecks for UAV positioning.

    Parameters:
    ground_users: List of ground user nodes.
    obstacles: List of objects that can obstruct line of sight.
    area_dimensions: Dictionary with keys 'xLength' and 'yLength' indicating the area size.
    min_altitude: Minimum altitude to consider for the heatmap.
    max_altitude: Maximum altitude to consider for the heatmap.
    weights: Dictionary with keys for GU, UAV, BS weights.
    sparsity_parameter: Controls the density of the heatmap by skipping certain points.
    target_user_indices: Optional list of indices specifying which ground users to include in the heatmap. If None, all users are considered.

    Returns:
    heatmap: A dictionary where each key is a coordinate tuple (x, y, z), and the value is a tuple (connection_score, GU_bottleneck).
    """
    x_length = area_dimensions['xLength']
    y_length = area_dimensions['yLength']
    heatmap = {}

    print("Calculating heatmap for GUs: "+str(target_user_indices))

    if target_user_indices is None:
        target_user_indices = range(len(ground_users))

    for altitude in range(min_altitude, max_altitude + 1):
        for x in range(0, x_length, sparsity_parameter):
            for y in range(0, y_length, sparsity_parameter):

                if is_position_inside_block(position=(x, y, altitude), blocks=obstacles):
                    continue

                connection_score = 0
                gu_bottleneck = float('inf')

                for user_index in target_user_indices:
                    user = ground_users[user_index]
                    viewpoint = Nodes([x, y, altitude])

                    if not path_is_blocked(obstacles, viewpoint, user):
                        connection_score += weights['GU']  # Increment connection score based on GU weight
                        data_rate = calculate_data_rate(uav_info, viewpoint.position, user.position, False)
                        gu_bottleneck = min(gu_bottleneck, data_rate)  # Update bottleneck with minimum data rate

                heatmap[(x, y, altitude)] = (connection_score, gu_bottleneck)

    return heatmap

def select_optimal_uav_position(heatmap, uncovered_gu_indices, ground_users, blocks):
    """
    Select the best UAV position based on the generated 3D heatmap.

    Parameters:
    heatmap: Dictionary generated by generate_3D_heatmap with positions and their scores.
    uncovered_gu_indices: List of indices of ground users that are not yet covered by any UAV.

    Returns:
    best_position: The coordinate (x, y, z) of the optimal UAV position.
    updated_uncovered_gu_indices: Updated list of uncovered ground user indices after placing the UAV.
    """
    best_position = None
    max_score = -float('inf')
    best_gu_bottleneck = float('inf')

    for position, (connection_score, gu_bottleneck) in heatmap.items():
        if connection_score > max_score or (connection_score == max_score and gu_bottleneck > best_gu_bottleneck):
            best_position = position
            max_score = connection_score
            # best_gu_bottleneck = min(gu_bottleneck, best_gu_bottleneck)
            best_gu_bottleneck = gu_bottleneck
        
    # updated_uncovered_gu_indices = [index for index in uncovered_gu_indices if not path_is_blocked(blocks, Nodes(best_position), ground_users[index])]
    updated_uncovered_gu_indices = [index for index in uncovered_gu_indices if path_is_blocked(blocks, Nodes(best_position), ground_users[index])]

    print("New UAV position is found: "+str(best_position)+" whose connection score is: "+str(max_score)+", with bottleneck: "+str(gu_bottleneck))

    return best_position, updated_uncovered_gu_indices

def find_optimal_uav_positions(ground_users, uavs, clustering_epsilon, min_cluster_size, obstacles, area_info, min_altitude, max_altitude, uav_info, weights, sparsity_parameter=1, print_para=False, print_prog=False):
    """
    Determine optimal positions for UAVs to maximize coverage of ground users.

    Parameters:
    ground_users: List of ground user nodes.
    uavs: List of UAV nodes.
    clustering_epsilon: Maximum distance between two points for DBSCAN clustering.
    min_cluster_size: Minimum number of points required to form a cluster in DBSCAN.
    obstacles: List of obstacles that can obstruct line of sight.
    area_info: Dictionary with keys 'xLength' and 'yLength' defining the simulation area dimensions.
    min_altitude: Minimum altitude to consider for UAV deployment.
    max_altitude: Maximum altitude to consider for UAV deployment.
    uav_info: Data structure containing information about UAVs, such as data rate and communication range.
    weights: Dictionary with keys for GU, UAV, BS weights.
    sparsity_parameter: Controls the density of the heatmap by skipping certain points.
    print_para: Boolean to print parameters for debugging.
    print_prog: Boolean to print progress updates.

    Returns:
    final_uav_positions: A list of UAV positions that maximize coverage of ground users.
    """
    uncovered_gu_indices = list(range(len(ground_users)))
    uav_positions = []

    while uncovered_gu_indices:
        print("Covering GUs...")
        heatmap = generate_3D_heatmap(ground_users, obstacles, area_info, min_altitude, max_altitude, weights, uav_info ,sparsity_parameter, target_user_indices=uncovered_gu_indices)
        best_position, uncovered_gu_indices = select_optimal_uav_position(heatmap, uncovered_gu_indices, ground_users, obstacles)
        uav_positions.append(best_position)

        print("There are uncovered GUs: "+str(uncovered_gu_indices))


    # After covering all GUs, optimize load distribution using DBSCAN
    for uav_position in uav_positions:
        print("Optimizing GUs...")
        gu_indices_for_this_uav = [index for index in range(len(ground_users)) if path_is_blocked(obstacles, Nodes(uav_position), ground_users[index]) == False]

        if len(gu_indices_for_this_uav) >= min_cluster_size:
            clustering = DBSCAN(eps=clustering_epsilon, min_samples=min_cluster_size).fit([ground_users[index].position for index in gu_indices_for_this_uav])
            cluster_labels = clustering.labels_

            for cluster in set(cluster_labels):
                if cluster == -1:
                    continue  # Skip noise points
                cluster_indices = [gu_indices_for_this_uav[i] for i in range(len(cluster_labels)) if cluster_labels[i] == cluster]
                cluster_heatmap = generate_3D_heatmap(ground_users, obstacles, area_info, min_altitude, max_altitude, weights, uav_info ,sparsity_parameter, target_user_indices=cluster_indices)
                best_cluster_position, _ = select_optimal_uav_position(cluster_heatmap, cluster_indices, ground_users, obstacles)
                uav_positions.append(best_cluster_position)

    return uav_positions
