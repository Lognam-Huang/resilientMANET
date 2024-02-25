import numpy as np
import json

from functions.path_is_blocked import path_is_blocked
from functions.scene_visualization import *
from classes.Nodes import Nodes
from functions.print_nodes import *

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
                    user = ground_users[index]
                    # Check if the line of sight from the current position to the ground user is blocked
                    curPos = Nodes([x,y,z])
                    if not path_is_blocked(blocks, curPos, user):
                        # If not blocked, increase the score for that point
                        heatmap[x, y, z - min_height] += 1
    return heatmap


def find_center_of_max_values(heatmap):
    # Step 1: Determine the maximum value
    max_value = np.max(heatmap)
    
    # Step 2: Find all coordinates of the maximum value
    max_positions = np.argwhere(heatmap == max_value)
    
    # Step 3: Calculate the center of these coordinates
    center_of_max_positions = np.mean(max_positions, axis=0)
    
    # Step 4: Find the coordinate closest to the center
    # Calculate the Euclidean distance from each maximum value coordinate to the center coordinate
    distances = np.linalg.norm(max_positions - center_of_max_positions, axis=1)
    # Find the index of the minimum distance
    closest_index = np.argmin(distances)
    # Return the coordinate closest to the center
    closest_position = max_positions[closest_index]
    
    return tuple(closest_position)


def update_considered_GU(ground_users, considered_GU, new_UAV_position, blocks):
    # Convert the new UAV position into a Nodes object to be compatible with the path_is_blocked function
    new_UAV_node = Nodes(new_UAV_position)
    
    # Create a list to store the indices of users that still need to be considered
    updated_considered_GU = considered_GU.copy()
    
    # Iterate through each index in considered_GU
    for index in considered_GU:
        user = ground_users[index]
        # Check for a line-of-sight connection
        if not path_is_blocked(blocks, new_UAV_node, user):
            # If a LOS connection exists, remove the user's index from the list
            updated_considered_GU.remove(index)
    
    return updated_considered_GU


def find_UAV_positions(ground_users, max_UAV_positions, blocks, scene, min_height, max_height):   

    # List to keep track of ground users that have not yet been covered by a UAV position
    considered_GU = list(range(len(ground_users)))
    # List to store the found UAV positions
    UAV_positions = []  
    
    for i in range(max_UAV_positions):
        # print(i)

        # Generate a heatmap for the current set of considered ground users
        heatmap = get_3D_heatmap(ground_users, blocks, scene, min_height, max_height, considered_GU)
        # Find the best UAV position based on the heatmap
        UAV_position = find_center_of_max_values(heatmap)

        # print(UAV_position)
        
        if UAV_position:  # Ensure a valid UAV position was found
            UAV_positions.append(UAV_position)
            # Update the list of considered ground users based on the new UAV position
            considered_GU = update_considered_GU(ground_users, considered_GU, UAV_position, blocks)

            # Optional: Visualize the 2D combined heatmap (commented out)
            visualize_2D_heatmap_combined(heatmap=heatmap, min_height=min_height, max_height= max_height)
            if not considered_GU:  # If all ground users are covered, end the loop early
                break
        else:
            break  # Stop the iteration if no new UAV position is found
    
    return UAV_positions

from sklearn.cluster import KMeans
def find_UAV_positions_kmeans(ground_users, max_UAV_positions, blocks, scene, min_height, max_height):
    considered_GU = list(range(len(ground_users)))  # Initially consider all ground users
    UAV_positions = []  # To store the UAV positions found

    # Convert ground_users to an array for KMeans
    # ground_users_array = np.array([user.position for user in ground_users if user.id in considered_GU])

    # print(considered_GU)
    # print_nodes(ground_users, True)

    ground_users_array = np.array([user.position for user in ground_users])

    # print(ground_users_array)

    # Apply KMeans clustering
    if len(ground_users_array) > 0 and max_UAV_positions > 0:
        kmeans = KMeans(n_clusters=min(max_UAV_positions, len(ground_users_array)), random_state=0).fit(ground_users_array)
        UAV_positions = kmeans.cluster_centers_.tolist()  # Use cluster centers as UAV positions
        
        # Here you would update considered_GU based on the new UAV positions
        # This step is left as an exercise because it requires integrating the update logic with your specific scenario
        # For example, you might need to check LOS from each new UAV position to ground users and update considered_GU accordingly

    return UAV_positions

from sklearn.cluster import DBSCAN

def find_UAV_positions_dbscan(ground_users, eps, min_samples, blocks, scene, min_height, max_height):
    considered_GU = [user.position for user in ground_users]  # Assume each user has a .position attribute
    UAV_positions = []  # To store the UAV positions found

    # Convert ground_users to an array for DBSCAN
    ground_users_array = np.array(considered_GU)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(ground_users_array)
    labels = dbscan.labels_

    # Find the unique clusters, ignoring noise if present (-1 label)
    unique_labels = set(labels) - {-1}

    for k in unique_labels:
        class_member_mask = (labels == k)
        xy = ground_users_array[class_member_mask]
        
        # For UAV position, we use the geometric center of each cluster
        UAV_position = xy.mean(axis=0)
        UAV_positions.append(UAV_position.tolist())

        # Here you might want to update considered_GU based on the new UAV positions
        # This requires integrating the update logic with your specific scenario
        
    return UAV_positions


