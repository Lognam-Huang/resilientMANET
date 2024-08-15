import numpy as np
from sklearn.cluster import DBSCAN

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

# def generate_3D_heatmap(ground_users, scene_data, weights, sparsity_parameter=1, target_user_indices=None):
#     """
#     Generate a 3D heatmap of connection scores and GU bottlenecks for UAV positioning.

#     Parameters:
 
#     Returns:
#     heatmap: A dictionary where each key is a coordinate tuple (x, y, z), and the value is a tuple (connection_score, GU_bottleneck).
#     """
#     x_length = scene_data['scenario']['xLength']
#     y_length = scene_data['scenario']['yLength']
#     heatmap = {}

#     print("Calculating heatmap for GUs: "+str(target_user_indices))

#     if target_user_indices is None:
#         target_user_indices = range(len(ground_users))

#     for altitude in range(scene_data['UAV']['min_height'], scene_data['UAV']['max_height'] + 1, sparsity_parameter):
#         for x in range(0, x_length, sparsity_parameter):
#             for y in range(0, y_length, sparsity_parameter):

#                 if is_position_inside_block(position=(x, y, altitude), blocks=scene_data['blocks']):
#                     continue

#                 connection_score = 0
#                 gu_bottleneck = float('inf')

#                 for user_index in target_user_indices:
#                     user = ground_users[user_index]
#                     viewpoint = Nodes([x, y, altitude])

#                     if not path_is_blocked(scene_data['blocks'], viewpoint, user):
#                         connection_score += weights['GU']  # Increment connection score based on GU weight
#                         data_rate = calculate_data_rate(scene_data['UAV'], viewpoint.position, user.position, False)
#                         gu_bottleneck = min(gu_bottleneck, data_rate)  # Update bottleneck with minimum data rate

#                         # print(data_rate)
#                         # print(gu_bottleneck)

#                 if connection_score > 0:
#                     heatmap[(x, y, altitude)] = (connection_score, gu_bottleneck)
#                 else:
#                     heatmap[(x, y, altitude)] = (connection_score, 10)
                

#     return heatmap

# def generate_3D_heatmap(ground_users, scene_data, weights, sparsity_parameter=1, target_user_indices=None, existing_uav_positions=None):
#     """
#     Generate a 3D heatmap of connection scores and GU bottlenecks for UAV positioning.

#     Parameters:
#     ground_users: List of ground user nodes.
#     scene_data: Dictionary containing scenario information (e.g., obstacles, BS locations).
#     weights: Dictionary containing weights for GU, UAV, and BS.
#     sparsity_parameter: Controls the density of the heatmap by skipping certain points.
#     target_user_indices: List of indices specifying which ground users to include in the heatmap. If None, all users are considered.
#     existing_uav_positions: List of positions of already placed UAVs.

#     Returns:
#     heatmap: A dictionary where each key is a coordinate tuple (x, y, z), and the value is a tuple (connection_score, GU_bottleneck).
#     """
#     x_length = scene_data['scenario']['xLength']
#     y_length = scene_data['scenario']['yLength']
#     heatmap = {}

#     print("Calculating heatmap for GUs: " + str(target_user_indices))

#     if target_user_indices is None:
#         target_user_indices = range(len(ground_users))

#     for altitude in range(scene_data['UAV']['min_height'], scene_data['UAV']['max_height'] + 1, sparsity_parameter):
#         for x in range(0, x_length, sparsity_parameter):
#             for y in range(0, y_length, sparsity_parameter):

#                 if is_position_inside_block(position=(x, y, altitude), blocks=scene_data['blocks']):
#                     continue

#                 connection_score = 0
#                 gu_bottleneck = float('inf')
#                 viewpoint = Nodes([x, y, altitude])

#                 # Calculate connection score and bottleneck for GUs
#                 for user_index in target_user_indices:
#                     user = ground_users[user_index]
#                     if not path_is_blocked(scene_data['blocks'], viewpoint, user):
#                         connection_score += weights['GU']  # Increment connection score based on GU weight
#                         data_rate = calculate_data_rate(scene_data['UAV'], viewpoint.position, user.position, False)
#                         gu_bottleneck = min(gu_bottleneck, data_rate)  # Update bottleneck with minimum data rate

#                 # Calculate connection score for existing UAVs (excluding the one being optimized)
#                 if existing_uav_positions:
#                     for uav_position in existing_uav_positions:
#                         if not path_is_blocked(scene_data['blocks'], viewpoint, Nodes(uav_position)):
#                             connection_score += weights['UAV']  # Increment connection score based on UAV weight

#                 # Calculate connection score for BS (Base Stations)
#                 if 'BS' in scene_data:
#                     for bs_position in scene_data['BS']:
#                         if not path_is_blocked(scene_data['blocks'], viewpoint, Nodes(bs_position)):
#                             connection_score += weights['BS']  # Increment connection score based on BS weight

#                 if connection_score > 0:
#                     heatmap[(x, y, altitude)] = (connection_score, gu_bottleneck)
#                 else:
#                     heatmap[(x, y, altitude)] = (connection_score, 10)

#     return heatmap

def generate_3D_heatmap(ground_users, scene_data, weights, sparsity_parameter=1, target_user_indices=None, existing_uav_positions=None, optimized_uav_index=-1):
    """
    Generate a 3D heatmap of connection scores and GU bottlenecks for UAV positioning.

    Parameters:
    ground_users: List of ground user nodes.
    scene_data: Dictionary containing scenario information (e.g., obstacles, BS locations).
    weights: Dictionary containing weights for GU, UAV, and BS.
    sparsity_parameter: Controls the density of the heatmap by skipping certain points.
    target_user_indices: List of indices specifying which ground users to include in the heatmap. If None, all users are considered.
    existing_uav_positions: List of positions of already placed UAVs.
    optimized_uav_index: Index of the UAV being optimized; -1 if no specific UAV is being optimized.

    Returns:
    heatmap: A dictionary where each key is a coordinate tuple (x, y, z), and the value is a tuple (connection_score, GU_bottleneck).
    """
    x_length = scene_data['scenario']['xLength']
    y_length = scene_data['scenario']['yLength']
    heatmap = {}

    print("Calculating heatmap for GUs: " + str(target_user_indices))

    if target_user_indices is None:
        target_user_indices = range(len(ground_users))

    for altitude in range(scene_data['UAV']['min_height'], scene_data['UAV']['max_height'] + 1, sparsity_parameter):
        for x in range(0, x_length, sparsity_parameter):
            for y in range(0, y_length, sparsity_parameter):

                if is_position_inside_block(position=(x, y, altitude), blocks=scene_data['blocks']):
                    continue

                connection_score = 0
                gu_bottleneck = float('inf')
                viewpoint = Nodes([x, y, altitude])

                # Calculate connection score and bottleneck for GUs
                for user_index in target_user_indices:
                    user = ground_users[user_index]
                    if not path_is_blocked(scene_data['blocks'], viewpoint, user):
                        connection_score += weights['GU']  # Increment connection score based on GU weight
                        data_rate = calculate_data_rate(scene_data['UAV'], viewpoint.position, user.position, False)
                        gu_bottleneck = min(gu_bottleneck, data_rate)  # Update bottleneck with minimum data rate

                # Calculate connection score for existing UAVs (excluding the one being optimized)
                if existing_uav_positions:
                    for uav_index, uav_position in enumerate(existing_uav_positions):
                        if uav_index == optimized_uav_index:
                            continue  # Skip the UAV that is being optimized
                        if not path_is_blocked(scene_data['blocks'], viewpoint, Nodes(uav_position)):
                            connection_score += weights['UAV']  # Increment connection score based on UAV weight

                # Calculate connection score for BS (Base Stations)
                if 'BS' in scene_data:
                    for bs_position in scene_data['BS']:
                        if not path_is_blocked(scene_data['blocks'], viewpoint, Nodes(bs_position)):
                            connection_score += weights['BS']  # Increment connection score based on BS weight

                if connection_score > 0:
                    heatmap[(x, y, altitude)] = (connection_score, gu_bottleneck)
                else:
                    heatmap[(x, y, altitude)] = (connection_score, 10)

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

    # print(heatmap)

    for position, (connection_score, gu_bottleneck) in heatmap.items():
        if connection_score > max_score or (connection_score == max_score and gu_bottleneck > best_gu_bottleneck):
            best_position = position
            max_score = connection_score
            # best_gu_bottleneck = min(gu_bottleneck, best_gu_bottleneck)
            best_gu_bottleneck = gu_bottleneck

            # print("New position information: position: "+str(best_position)+", connection score: "+str(max_score)+", GU bottleneck: "+str(best_gu_bottleneck))
        
    # updated_uncovered_gu_indices = [index for index in uncovered_gu_indices if not path_is_blocked(blocks, Nodes(best_position), ground_users[index])]
    updated_uncovered_gu_indices = [index for index in uncovered_gu_indices if path_is_blocked(blocks, Nodes(best_position), ground_users[index])]

    print("New UAV position is found: "+str(best_position)+" whose connection score is: "+str(max_score)+", with bottleneck: "+str(best_gu_bottleneck))

    return best_position, updated_uncovered_gu_indices

from scipy.cluster.hierarchy import linkage, fcluster

def find_two_clusters_hierarchical(gu_positions_for_max_uav):
    """
    Use hierarchical clustering to find exactly two clusters.

    Parameters:
    gu_positions_for_max_uav: List of positions of GUs connected to the max load UAV.

    Returns:
    clusters: Dictionary with two keys, each containing a list of indices for GUs in each cluster.
    """
    Z = linkage(gu_positions_for_max_uav, method='ward')  # 'ward' minimizes the variance of the clusters
    cluster_labels = fcluster(Z, 2, criterion='maxclust')  # Force exactly 2 clusters

    clusters = {0: [], 1: []}
    for i, label in enumerate(cluster_labels):
        clusters[label-1].append(i)

    # Check if any cluster is empty or has very few GUs
    if len(clusters[0]) == 0 or len(clusters[1]) == 0:
        print("Warning: One of the clusters is empty or has too few GUs. Rebalancing...")
        total_gus = len(gu_positions_for_max_uav)
        if len(clusters[0]) == 0:
            clusters[0], clusters[1] = clusters[1][:total_gus//2], clusters[1][total_gus//2:]
        elif len(clusters[1]) == 0:
            clusters[1], clusters[0] = clusters[0][:total_gus//2], clusters[0][total_gus//2:]
    
    return clusters

def find_optimal_uav_positions(ground_users, uavs, clustering_epsilon, min_cluster_size, scene_data, weights, sparsity_parameter=1, print_para=False, print_prog=False):
    uncovered_gu_indices = list(range(len(ground_users)))
    uav_positions = []
    available_uav_indices = list(range(len(uavs)))  # Track available UAVs by their indices

    while available_uav_indices:
        print("Available UAVs: " + str(available_uav_indices))
        if uncovered_gu_indices:
            print("Covering GUs...")
            # heatmap = generate_3D_heatmap(ground_users, scene_data, weights, sparsity_parameter, target_user_indices=uncovered_gu_indices)
            heatmap = generate_3D_heatmap(ground_users, scene_data, weights, sparsity_parameter, target_user_indices=uncovered_gu_indices, existing_uav_positions=uav_positions)

            best_position, uncovered_gu_indices = select_optimal_uav_position(heatmap, uncovered_gu_indices, ground_users, scene_data['blocks'])
            uav_positions.append(best_position)

            # Remove the used UAV from the available list
            available_uav_indices.pop(0)  # Assuming you assign UAVs in order, adjust this if necessary

            print("There are uncovered GUs: " + str(uncovered_gu_indices))
        else:
            # After covering all GUs, optimize load distribution using Hierarchical Clustering
            print("Optimizing GUs...")

            # Step 1: Calculate which UAV each GU prefers based on the maximum data rate
            gu_to_uav_map = {uav_index: [] for uav_index in range(len(uav_positions))}

            for gu_index in range(len(ground_users)):
                best_uav = None
                max_data_rate = -float('inf')

                for uav_index, uav_position in enumerate(uav_positions):
                    data_rate = calculate_data_rate(scene_data['UAV'], uav_position, ground_users[gu_index].position, path_is_blocked(scene_data['blocks'], Nodes(uav_position), ground_users[gu_index]))
                    if data_rate > max_data_rate:
                        max_data_rate = data_rate
                        best_uav = uav_index

                if best_uav is not None:
                    gu_to_uav_map[best_uav].append(gu_index)

            # Step 2: Find the UAV with the maximum load (most GUs connected)
            max_load_uav = max(gu_to_uav_map, key=lambda k: len(gu_to_uav_map[k]))
            gu_indices_for_max_uav = gu_to_uav_map[max_load_uav]

            print("Optimized UAV is: " + str(max_load_uav) + ", with covered GUs: " + str(gu_indices_for_max_uav))

            # Step 3: Apply Hierarchical Clustering to split GUs into two clusters
            gu_positions_for_max_uav = [ground_users[index].position for index in gu_indices_for_max_uav]
            clusters = find_two_clusters_hierarchical(gu_positions_for_max_uav)

            print("Hierarchical clustering is applied, and we find 2 clusters.")

            # Step 4: Find new positions for the UAVs based on the clusters
            new_positions = []
            for cluster in clusters.values():
                if cluster:  # Only generate a new position if the cluster is not empty
                    # cluster_heatmap = generate_3D_heatmap(ground_users, scene_data, weights, sparsity_parameter, target_user_indices=[gu_indices_for_max_uav[i] for i in cluster])
                    # cluster_heatmap = generate_3D_heatmap(ground_users, scene_data, weights, sparsity_parameter, target_user_indices=[gu_indices_for_max_uav[i] for i in cluster], existing_uav_positions=uav_positions)

                    # Generate heatmap considering all UAVs except the one being optimized
                    cluster_heatmap = generate_3D_heatmap(
                        ground_users, 
                        scene_data, 
                        weights, 
                        sparsity_parameter, 
                        target_user_indices=[gu_indices_for_max_uav[i] for i in cluster],
                        existing_uav_positions=uav_positions, 
                        optimized_uav_index=max_load_uav  # Skip the UAV that is currently being optimized
                    )


                    best_cluster_position, _ = select_optimal_uav_position(cluster_heatmap, [gu_indices_for_max_uav[i] for i in cluster], ground_users, scene_data['blocks'])
                    new_positions.append(best_cluster_position)

            if len(new_positions) == 2:
                print("Optimization done, found UAV is updated from " + str(uav_positions[max_load_uav]) + " to " + str(new_positions[0]) + ". Meanwhile the new UAV will locate at: " + str(new_positions[1]))
                # Update the position of the overloaded UAV
                uav_positions[max_load_uav] = new_positions[0]

                # Add a new UAV position for the second cluster
                uav_positions.append(new_positions[1])
                available_uav_indices.pop(0)
            else:
                print("Error: Clustering resulted in an empty cluster. Skipping this optimization.")
                break  # Exit the loop if optimization fails due to empty clusters

            
        for uav_index, uav_position in enumerate(uav_positions):
            uavs[uav_index].set_position(uav_position)

    return uav_positions

