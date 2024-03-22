import numpy as np
import json

from functions.path_is_blocked import path_is_blocked
from functions.scene_visualization import *
from classes.Nodes import Nodes
from functions.print_nodes import *

from functions.calculate_data_rate import *

def get_3D_heatmap(ground_users, blocks, scene_info, min_height, max_height, considered_users_indices=None):
    x_length = scene_info['xLength']
    y_length = scene_info['yLength']
    # Initialize a three-dimensional array to store the score for each point
    heatmap = np.zeros((x_length, y_length, max_height - min_height + 1))
    
    # If considered_users_indices is not provided, consider all users
    if considered_users_indices is None:
        considered_users_indices = range(len(ground_users))
    
    # Iterate through each height
    for z in range(min_height, max_height + 1):
        # Iterate through each x and y coordinate
        for x in range(x_length):
            for y in range(y_length):
                # Only consider specified users
                for index in considered_users_indices:
                    # user = ground_users[index]

                    # since indices starts with 1, we need to consider -1
                    user = ground_users[index-1]

                    # Check if the line of sight from the current position to the ground user is blocked
                    curPos = Nodes([x,y,z])
                    if not path_is_blocked(blocks, curPos, user):
                        # If not blocked, increase the score for that point
                        heatmap[x, y, z - min_height] += 1
    return heatmap





from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

def get_max_cluster_point(ground_users, UAVs, eps, min_samples, blocks, scene, min_height, max_height, UAVInfo):
    considered_GU = list(range(len(ground_users)))
    considered_UAVs = list(range(len(UAVs)))
    heatmap = get_3D_heatmap(ground_users, blocks, scene, min_height, max_height, considered_GU)

    max_capacities_tracks = [find_max_capacity_for_each_gu(ground_users, UAVs, blocks, UAVInfo)]

    while len(considered_UAVs) > 0:
        
        # Existing logic to find and position UAVs
        # ...
        # Step 1: Find all coordinates of the maximum values in the heatmap
        max_value = np.max(heatmap)
        max_points = np.argwhere(heatmap == max_value)
        
        # Step 2: Use DBSCAN to cluster the points of maximum values
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(max_points)
        labels = clustering.labels_
        
        # Step 3: Find the largest cluster
        # Count the number of points in each cluster (excluding noise, if any)
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(counts) == 0:  # In case all points are considered noise by DBSCAN
            return None
        largest_cluster_label = unique_labels[np.argmax(counts)]

        
        
        # Step 4: Randomly select a point from the largest cluster
        largest_cluster_points = max_points[labels == largest_cluster_label]

        if len(considered_GU) > 0:
            selected_point = largest_cluster_points[np.random.choice(largest_cluster_points.shape[0]), :]

            selected_point[2] = selected_point[2] + min_height
            print("UAV node atttemps to go to:")
            # print(selected_point[0])
            print(selected_point)
            # print(type(selected_point))

            curUAV = considered_UAVs[0]
            # UAVs[curUAV].set_position(selected_point)

            # print("UAV height should be:")
            # print(selected_point[2] + min_height)

            UAVs[curUAV].set_position((selected_point[0], selected_point[1], selected_point[2]))
            # UAVs[curUAV].set_position((10,24,1))

            considered_UAVs.remove(curUAV)
            

            #update uavs and gus
            # print(considered_GU)
            # print("ss")

            considered_GU, disconnected_GU_number = update_considered_GU(ground_users, UAVs, max_height, blocks)
            print("Considered GU, which means GU that still disconnected are:")
            print(considered_GU)
            print(disconnected_GU_number)

            #update heatmap
            heatmap = get_3D_heatmap(ground_users, blocks, scene, min_height, max_height, considered_GU)

            max_capacities_tracks.append(find_max_capacity_for_each_gu(ground_users, UAVs, blocks, UAVInfo))
        else:
            if largest_cluster_points.shape[0] < 2:
                # return None  # Not enough points to select two distinct points
                break
            
            # Compute pairwise distances and find the furthest points
            from scipy.spatial.distance import pdist, squareform
            distances = squareform(pdist(largest_cluster_points, 'euclidean'))
            i, j = np.unravel_index(np.argmax(distances), distances.shape)

            # Assume that we are selecting two UAVs for these points
            selected_points = largest_cluster_points[[i, j], :]
            for point, curUAV in zip(selected_points, considered_UAVs[:2]):
                point[2] += min_height
                print(f"UAV {curUAV} attempts to go to: {point}")
                UAVs[curUAV].set_position((point[0], point[1], point[2]))
                considered_UAVs.remove(curUAV)

            # Update the heatmap and capacities based on the new UAV positions
            considered_GU, disconnected_GU_number = update_considered_GU(ground_users, UAVs, max_height, blocks)
            heatmap = get_3D_heatmap(ground_users, blocks, scene, min_height, max_height, considered_GU)
            max_capacities_tracks.append(find_max_capacity_for_each_gu(ground_users, UAVs, blocks, UAVInfo))


    return max_capacities_tracks



def update_considered_GU(ground_user, UAVs, max_height, blocks):
    updated_considered_GU = list()
    GU_counter = 1

    disconnected_GU_number = len(ground_user)

    for gu in ground_user:
        cur_gu_is_connected = False
        for uav in UAVs:
            # print(uav.position)
            # print(uav.position[2])
            # print(max_height)

            if uav.position[2] > max_height:
                continue
            # print(uav.position)
            # print(uav.position[2])
            # print(max_height)
            # if not path_is_blocked(blocks, uav.position, gu.position):
            if not path_is_blocked(blocks, uav, gu):
                cur_gu_is_connected = True

                disconnected_GU_number -= 1

                # print("This gu is connected to an UAV")
                break
        
        if not cur_gu_is_connected:
            updated_considered_GU.append(GU_counter)
            # print("SA")

        # if cur_gu_is_connected:
        #     updated_considered_GU.append(GU_counter)
        #     # print("WA")
        
        GU_counter += 1
    
    return updated_considered_GU, disconnected_GU_number

def find_max_capacity_for_each_gu(ground_users, UAVs, blocks, UAVInfo):
    # 用于存储每个地面用户的最大capacity
    max_capacities = {}

    for gu in ground_users:
        # 初始化当前GU的最大capacity为0
        max_capacity = 0

        for uav in UAVs:
            # 检查路径是否被阻塞
            blocked = path_is_blocked(blocks, uav, gu)
            # 计算data rate，考虑阻塞情况
            data_rate = calculate_data_rate(UAVInfo, uav.position, gu.position, blocked)
            
            if max_capacity < data_rate:

                # 更新最大capacity
                max_capacity = max(max_capacity, data_rate)
                gu.set_connection(uav.node_number)

            # print(max_capacity)

        # 记录当前GU的最大capacity
        max_capacities[gu] = max_capacity

    return max_capacities

def find_most_connected_ground_users(nodes_list):
    """
    Finds the most frequently connected node among ground users and returns a list
    of node_numbers of ground users connected to this node.

    Parameters:
    nodes_list (list): A list of Nodes objects.

    Returns:
    list: A list of node_numbers for ground users connected to the most frequent node.
    """
    # Filter out the ground users from the nodes list
    ground_users = [node for node in nodes_list if node.type == "ground users"]

    # Create a dictionary to count the occurrences of connected nodes
    connection_counts = {}
    for user in ground_users:
        for connected_node in user.connected_nodes:
            if connected_node in connection_counts:
                connection_counts[connected_node] += 1
            else:
                connection_counts[connected_node] = 1

    # Find the node that has the highest number of connections
    if connection_counts:
        most_common_node = max(connection_counts, key=connection_counts.get)
    else:
        return []

    # Collect the node_numbers of ground users connected to the most common node
    connected_user_numbers = [user.node_number for user in ground_users if most_common_node in user.connected_nodes]

    return connected_user_numbers

def find_distant_points_in_max_cluster(heatmap, eps, min_samples):
    # Step 1: Find all coordinates of the maximum values in the heatmap
    max_value = np.max(heatmap)
    max_points = np.argwhere(heatmap == max_value)

    # Step 2: Use DBSCAN to cluster the points of maximum values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(max_points)
    labels = clustering.labels_

    # Step 3: Find the largest cluster
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(counts) == 0:  # In case all points are considered noise by DBSCAN
        return None
    largest_cluster_label = unique_labels[np.argmax(counts)]

    # Step 4: Find two points in the largest cluster that are as far apart as possible
    largest_cluster_points = max_points[labels == largest_cluster_label]
    
    if largest_cluster_points.shape[0] < 2:
        return None  # Not enough points to select two distinct points

    # Calculate the pairwise distances between all points in the largest cluster
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(largest_cluster_points, 'euclidean'))

    # Find the pair with the maximum distance
    i, j = np.unravel_index(distances.argmax(), distances.shape)

    selected_points = largest_cluster_points[[i, j], :]

    return selected_points