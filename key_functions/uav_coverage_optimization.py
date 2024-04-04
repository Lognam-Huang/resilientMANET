import numpy as np
from functions.path_is_blocked import path_is_blocked
from classes.Nodes import Nodes
from sklearn.cluster import DBSCAN

from functions.calculate_data_rate import *
from scipy.spatial.distance import pdist, squareform

from functions.print_nodes import *

def generate_visibility_heatmap(ground_users, obstacles, area_dimensions, min_altitude, max_altitude, target_user_indices=None):
    """
    Generate a 3D heatmap of visibility for ground users within a specified area and altitude range.

    Parameters:
    ground_users: List of ground user nodes.
    obstacles: List of objects that can obstruct line of sight.
    area_dimensions: Dictionary with keys 'xLength' and 'yLength' indicating the area size.
    min_altitude: Minimum altitude to consider for the heatmap.
    max_altitude: Maximum altitude to consider for the heatmap.
    target_user_indices: Optional list of indices specifying which ground users to include in the heatmap. If None, all users are considered.

    Returns:
    A 3D numpy array where each element represents the number of ground users visible from that point.
    """
    x_length = area_dimensions['xLength']
    y_length = area_dimensions['yLength']
    heatmap = np.zeros((x_length, y_length, max_altitude - min_altitude + 1))

    if target_user_indices is None:
        target_user_indices = range(len(ground_users))

    for altitude in range(min_altitude, max_altitude + 1):
        for x in range(x_length):
            for y in range(y_length):
                for user_index in target_user_indices:
                    user = ground_users[user_index]  # Adjusting for 0-based index if necessary

                    # print("User index is:"+str(user_index))
                    # print("User number of GU is:")
                    # print_node_number(user)

                    viewpoint = Nodes([x, y, altitude])
                    if not path_is_blocked(obstacles, viewpoint, user):
                        heatmap[x, y, altitude - min_altitude] += 1

    return heatmap

def find_optimal_uav_positions(ground_users, uavs, clustering_epsilon, min_cluster_size, obstacles, area_info, min_altitude, max_altitude, uav_info):
    """
    Determine optimal positions for UAVs to maximize coverage of ground users by analyzing
    the density of ground users within the 3D space and clustering high-density areas.

    Parameters:
    ground_users: List of ground user nodes.
    uavs: List of UAV nodes.
    clustering_epsilon: The maximum distance between two points to be considered in the same neighborhood for DBSCAN clustering.
    min_cluster_size: The minimum number of points required to form a cluster in DBSCAN.
    obstacles: List of obstacles that can obstruct line of sight.
    area_info: Dictionary with keys 'xLength' and 'yLength' defining the simulation area dimensions.
    min_altitude, max_altitude: The altitude range to consider for UAV deployment.
    uav_info: Data structure containing information about UAVs, such as data rate and communication range.

    Returns:
    A list of dictionaries representing the maximum communication capacities for each ground user after positioning UAVs.
    """
    # Initialize indices for active ground users and UAVs
    active_ground_users_indices = list(range(len(ground_users)))
    active_uavs_indices = list(range(len(uavs)))
    max_capacity_records = []

    # Main loop to find optimal positions for UAVs
    while active_uavs_indices:
        # Generate visibility heatmap for current ground user positions
        heatmap = generate_visibility_heatmap(ground_users, obstacles, area_info, min_altitude, max_altitude, active_ground_users_indices)

        # Find maximum communication capacity for current UAV positions
        max_capacity_records.append(find_maximum_capacity_per_ground_user(ground_users, uavs, obstacles, uav_info, max_altitude))

        # Find the highest value in the heatmap and its points
        max_value = np.max(heatmap)
        max_value_points = np.argwhere(heatmap == max_value)

        # Perform DBSCAN clustering on points with the highest visibility value
        clustering = DBSCAN(eps=clustering_epsilon, min_samples=min_cluster_size).fit(max_value_points)
        cluster_labels = clustering.labels_

        # Determine the largest cluster of points
        unique_labels, counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
        if counts.size == 0:
            print("All points are considered noise by DBSCAN")
            return None

        largest_cluster_label = unique_labels[np.argmax(counts)]
        largest_cluster_points = max_value_points[cluster_labels == largest_cluster_label]

        # Select points from the largest cluster for UAV positioning
        if len(largest_cluster_points) < 2:
            print("Not enough points in the cluster for further optimization")
            break

        selected_indices = np.random.choice(len(largest_cluster_points), size=2, replace=False)
        selected_points = largest_cluster_points[selected_indices]

        # Move or place UAVs to selected points from the cluster
        for i, point in enumerate(selected_points):
            if i < len(active_uavs_indices):
                uav_index = active_uavs_indices.pop(0)
                uavs[uav_index].set_position((point[0], point[1], point[2] + min_altitude))
                print(f"UAV at index {uav_index} moved to {uavs[uav_index].position}")

        # Update the connectivity and recalculate the heatmap and capacities after UAV movement
        active_ground_users_indices, disconnected_users_count = update_connected_ground_users(ground_users, uavs, max_altitude, obstacles)
        max_capacity_records.append(find_maximum_capacity_per_ground_user(ground_users, uavs, obstacles, uav_info, max_altitude))

        print("Current ground user coverage:")
        print(active_ground_users_indices)

    return max_capacity_records


def update_connected_ground_users(ground_users, uavs, max_altitude, obstacles):
    """
    Update the list of ground users considered for connectivity based on their line of sight to UAVs.

    Parameters:
    ground_users: List of ground user nodes.
    uavs: List of UAV nodes.
    max_altitude: The maximum altitude UAVs can fly to maintain connectivity.
    obstacles: List of obstacles that can block the line of sight between UAVs and ground users.

    Returns:
    Tuple containing the list of indices of ground users still considered for connectivity and the number of disconnected ground users.
    """
    considered_ground_users = []
    disconnected_users_count = len(ground_users)

    for index, gu in enumerate(ground_users, start=0):
    # for index, gu in enumerate(ground_users, start=1):
        # print("idx = ")
        # print(index)
        is_connected = False
        for uav in uavs:
            if uav.position[2] > max_altitude:
                continue

            if not path_is_blocked(obstacles, uav, gu):
                is_connected = True
                disconnected_users_count -= 1
                break

        if not is_connected:
            considered_ground_users.append(index)

    return considered_ground_users, disconnected_users_count

def find_maximum_capacity_per_ground_user(ground_users, uavs, obstacles, uav_info, max_altitude):
    """
    Calculate the maximum communication capacity for each ground user based on the line of sight to UAVs.

    Parameters:
    ground_users: List of ground user nodes.
    uavs: List of UAV nodes.
    obstacles: List of objects that can block the line of sight between ground users and UAVs.
    uav_info: Information about UAVs' capabilities used to calculate the data rate.

    Returns:
    Dictionary mapping each ground user to its maximum communication capacity.
    """
    maximum_capacities = {}

    # minimum_capacity is used for checking if there are enormous efficiency downhills while optimizing overload
    minimum_capacity = math.inf

    for ground_user in ground_users:
        max_capacity = 0  # Initialize the maximum capacity for the current ground user

        for uav in uavs:
            if uav.position[2] > max_altitude:
                continue
            
            is_blocked = path_is_blocked(obstacles, uav, ground_user)
            data_rate = calculate_data_rate(uav_info, uav.position, ground_user.position, is_blocked)
            
            # Update the maximum capacity if the current data rate is higher
            if data_rate > max_capacity:
                max_capacity = data_rate

                if not is_blocked:
                    ground_user.set_connection(uav.node_number)  # Store the connection to the UAV

        maximum_capacities[ground_user] = max_capacity

        minimum_capacity = min(minimum_capacity, max_capacity)        

    print("The worst capacity of all GU at the moment is:")
    print(minimum_capacity)
    
    return maximum_capacities

def find_most_connected_ground_users(nodes_list):
    """
    Finds the ground user node that is most frequently connected to others and returns the list
    of node numbers of ground users connected to this node.

    Parameters:
    - nodes_list (list): A list of Nodes objects, representing both ground users and other types of nodes.

    Returns:
    - list: A list of node numbers for ground users that are connected to the most frequently connected node.
    """
    # Extract only ground users from the nodes list
    # ground_users = [node for node in nodes_list if node.type == "ground user"]
    ground_users = nodes_list

    # Count the occurrences of each connected node among ground users
    connection_counts = {}
    for user in ground_users:
        for connected_node in user.connected_nodes:
            connection_counts[connected_node] = connection_counts.get(connected_node, 0) + 1

    # Determine the node with the highest number of connections
    most_common_node = max(connection_counts, key=connection_counts.get, default=None)

    print("Each UAV cover some GUs:")
    print(connection_counts)
    # print(most_common_node)

    if most_common_node == None:
        print("There is no frequently used UAV node")
        return []

    # Identify all ground users that are connected to this most common node
    most_connected_user_numbers = [user.node_number for user in ground_users if most_common_node in user.connected_nodes]

    # print(most_connected_user_numbers)

    return most_connected_user_numbers, most_common_node

# not used at the present
def find_furthest_points_in_largest_cluster(heatmap, epsilon, min_samples):
    """
    Identify two furthest apart points within the largest cluster of maximum value points in a heatmap using DBSCAN clustering.

    Parameters:
    heatmap: 2D or 3D numpy array representing the heatmap data.
    epsilon: The maximum distance between two points for them to be considered as part of the same cluster in DBSCAN.
    min_samples: The minimum number of samples in a neighborhood for a point to be considered as a core point in DBSCAN.

    Returns:
    A numpy array of the two furthest apart points in the largest cluster, or None if no such points can be found.
    """
    # Find all coordinates where the heatmap reaches its maximum value
    max_value = np.max(heatmap)
    max_value_points = np.argwhere(heatmap == max_value)

    # Cluster the points of maximum values using DBSCAN
    clustering = DBSCAN(eps=epsilon, min_samples=min_samples).fit(max_value_points)
    labels = clustering.labels_

    # Determine the largest cluster
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if not counts.size:
        return None  # Exit if all points are considered as noise

    largest_cluster_label = unique_labels[np.argmax(counts)]
    largest_cluster_points = max_value_points[labels == largest_cluster_label]

    if largest_cluster_points.shape[0] < 2:
        return None  # Exit if the largest cluster has fewer than two points

    # Compute pairwise distances between points in the largest cluster and find the furthest pair
    distances = squareform(pdist(largest_cluster_points, 'euclidean'))
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    furthest_points = largest_cluster_points[[i, j], :]

    return furthest_points

import matplotlib.pyplot as plt

def plot_gu_capacities(capacities_tracks):
    """
    Plot the maximum capacity for each ground user over time.
    """
    gu_capacities = {}

    for capacities_dict in capacities_tracks:
        for gu, capacity in capacities_dict.items():
            # gu_id = id(gu)
            gu_id = gu.node_number
            gu_capacities.setdefault(gu_id, []).append(capacity)

    plt.figure(figsize=(12, 8))
    for gu_id, capacities in gu_capacities.items():
        plt.plot(capacities, label=f'GU {gu_id}')

    plt.title('Max Capacity for each GU over time')
    plt.xlabel('Time/Scenario')
    plt.ylabel('Max Capacity')
    plt.legend()
    plt.show()