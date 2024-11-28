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
            # visualize_2D_heatmap_combined(heatmap=heatmap, min_height=min_height, max_height= max_height)
            
            if not considered_GU:  # If all ground users are covered, end the loop early
                break
        else:
            break  # Stop the iteration if no new UAV position is found
    
    return UAV_positions

# def find_UAV_positions_kmeans(ground_users, max_UAV_positions, blocks, scene, min_height, max_height):
#     considered_GU = list(range(len(ground_users)))  # Initially consider all ground users
#     UAV_positions = []  # To store the UAV positions found

#     # Convert ground_users to an array for KMeans
#     # ground_users_array = np.array([user.position for user in ground_users if user.id in considered_GU])

#     # print(considered_GU)
#     # print_nodes(ground_users, True)

#     ground_users_array = np.array([user.position for user in ground_users])

#     # print(ground_users_array)

#     # Apply KMeans clustering
#     if len(ground_users_array) > 0 and max_UAV_positions > 0:
#         kmeans = KMeans(n_clusters=min(max_UAV_positions, len(ground_users_array)), random_state=0).fit(ground_users_array)
#         UAV_positions = kmeans.cluster_centers_.tolist()  # Use cluster centers as UAV positions
        
#         # Here you would update considered_GU based on the new UAV positions
#         # This step is left as an exercise because it requires integrating the update logic with your specific scenario
#         # For example, you might need to check LOS from each new UAV position to ground users and update considered_GU accordingly

#     return UAV_positions

# def find_UAV_positions_dbscan(ground_users, eps, min_samples, blocks, scene, min_height, max_height):
    # considered_GU = [user.position for user in ground_users]  # Assume each user has a .position attribute
    # UAV_positions = []  # To store the UAV positions found

    # # Convert ground_users to an array for DBSCAN
    # ground_users_array = np.array(considered_GU)

    # # Apply DBSCAN clustering
    # dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(ground_users_array)
    # labels = dbscan.labels_

    # # Find the unique clusters, ignoring noise if present (-1 label)
    # unique_labels = set(labels) - {-1}

    # for k in unique_labels:
    #     class_member_mask = (labels == k)
    #     xy = ground_users_array[class_member_mask]
        
    #     # For UAV position, we use the geometric center of each cluster
    #     UAV_position = xy.mean(axis=0)
    #     UAV_positions.append(UAV_position.tolist())

    #     # Here you might want to update considered_GU based on the new UAV positions
    #     # This requires integrating the update logic with your specific scenario
        
    # return UAV_positions


# UAV_positions = find_UAV_positions(ground_users, max_UAV_positions, blocks, scene, min_height, max_height)
# print("Founded UAV positions by averaging: ", UAV_positions)

# UAV_positions_kmeans = find_UAV_positions_kmeans(ground_users, max_UAV_positions, blocks, scene, min_height, max_height)
# print("Founded UAV positions by KMeans: ", UAV_positions_kmeans)