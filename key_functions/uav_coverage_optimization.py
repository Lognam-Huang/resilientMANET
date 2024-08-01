import numpy as np
from functions.path_is_blocked import path_is_blocked
from classes.Nodes import Nodes
from sklearn.cluster import DBSCAN

from functions.calculate_data_rate import *
from scipy.spatial.distance import pdist, squareform

from functions.print_nodes import *

import time

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

                if is_position_inside_block(position=(x,y,altitude), blocks=obstacles):
                    continue

                for user_index in target_user_indices:
                    user = ground_users[user_index]  # Adjusting for 0-based index if necessary

                    # print("User index is:"+str(user_index))
                    # print("User number of GU is:")
                    # print_node_number(user)

                    viewpoint = Nodes([x, y, altitude])
                    if not path_is_blocked(obstacles, viewpoint, user):
                        heatmap[x, y, altitude - min_altitude] += 1

    return heatmap

def find_optimal_uav_positions(ground_users, uavs, clustering_epsilon, min_cluster_size, obstacles, area_info, min_altitude, max_altitude, uav_info, print_para=False, print_prog=False):
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
    altitude_range: Tuple of minimum and maximum altitudes to consider for UAV deployment.
    uav_info: Data structure containing information about UAVs, such as data rate and communication range.

    Returns:
    A list of dictionaries representing the maximum communication capacities for each ground user after positioning UAVs.
    """

    if print_para:
        print("find_optimal_uav_positions called with the following parameters:")
        print("Ground Users:")
        print_nodes(ground_users, onlyPosition=True)
        print("UAVs:")
        print_nodes(uavs, onlyPosition=True)
        print("Clustering Epsilon:", clustering_epsilon)
        print("Min Cluster Size:", min_cluster_size)
        print("Obstacles:", obstacles)
        print("Area Info:", area_info)
        print("Min Altitude:", min_altitude)
        print("Max Altitude:", max_altitude)
        print("UAV Info:", uav_info)

    active_ground_users_indices = list(range(len(ground_users)))
    active_uavs_indices = list(range(len(uavs)))
    max_capacity_records = []

    start_time = time.time()

    while active_uavs_indices:

        target_uav = None

        if not active_ground_users_indices:
            active_ground_users_indices, target_uav = find_most_connected_ground_users(ground_users)
        

        heatmap = generate_visibility_heatmap(ground_users, obstacles, area_info, min_altitude, max_altitude, active_ground_users_indices)
        max_capacity_records.append(find_maximum_capacity_per_ground_user(ground_users, uavs, obstacles, uav_info, max_altitude))

        max_value = np.max(heatmap)
        max_value_points = np.argwhere(heatmap == max_value)        
        clustering = DBSCAN(eps=clustering_epsilon, min_samples=min_cluster_size).fit(max_value_points)
        cluster_labels = clustering.labels_
        
        unique_labels, counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
        if counts.size == 0:
            return None  # Exit if all points are considered noise by DBSCAN
        
        largest_cluster_label = unique_labels[np.argmax(counts)]
        largest_cluster_points = max_value_points[cluster_labels == largest_cluster_label]

        if target_uav == None:
            selected_point = largest_cluster_points[np.random.choice(len(largest_cluster_points))]
            selected_point[2] += min_altitude  # Adjust altitude to absolute value

            # if not is_position_inside_block(selected_point, obstacles):
            #         # break
            #     continue

            current_uav = active_uavs_indices.pop(0)
            uavs[current_uav].set_position((selected_point[0], selected_point[1], selected_point[2]))

            # print("Trying to cover more GUs:")
            # print(str(selected_point))

            active_ground_users_indices, disconnected_users_count = update_connected_ground_users(ground_users, uavs, max_altitude, obstacles)
            # print(active_ground_users_indices)
        else:
            # print("Trying to optimize positions:")        
            # print("According to new heatmap, the max value is:")
            # print(np.max(heatmap))

            if len(largest_cluster_points) < 2:
                print("There is not enough points in the cluster for further optimization")

            selected_indices = np.random.choice(len(largest_cluster_points), size=2, replace=False)
            selected_points = largest_cluster_points[selected_indices]

            # if not is_position_inside_block(selected_points[0], obstacles) and not is_position_inside_block(selected_points[1], obstacles):
            #         # break
            #     continue
            
            # print("Modify existed UAV node at:")
            # print(uavs[target_uav].position)

            uavs[target_uav].set_position((selected_points[0][0],selected_points[0][1],selected_points[0][2]+min_altitude))

            # print("New position of existed UAV node is:")
            # print(uavs[target_uav].position)

            uav_index = active_uavs_indices[:1][0]
            uavs[uav_index].set_position((selected_points[1][0],selected_points[1][1],selected_points[1][2]+min_altitude))
            
            # print("Add a new UAV node for workload at:")
            # print(uavs[uav_index].position)

            active_uavs_indices.remove(uav_index)
            active_ground_users_indices, disconnected_users_count = update_connected_ground_users(ground_users, uavs, max_altitude, obstacles)
            
    # print("Check whether all GUs are covered:")
    # print(active_ground_users_indices)

    max_capacity_records.append(find_maximum_capacity_per_ground_user(ground_users, uavs, obstacles, uav_info, max_altitude))

    if print_prog:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The code block ran in {elapsed_time} seconds")

        print("All UAV positions:")
        for uav in uavs:
            print(f"UAV {uav.node_number} position: {uav.position}")
    
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

def find_maximum_capacity_per_ground_user(ground_users, uavs, obstacles, uav_info, max_altitude, print_prog=False):
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
        target_uav_id = None

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
                    target_uav_id = uav.node_number

        maximum_capacities[ground_user] = max_capacity, target_uav_id

        minimum_capacity = min(minimum_capacity, max_capacity)        

    if print_prog:
        print("The worst capacity of all GU at the moment is:")
        print(minimum_capacity)
    
    return maximum_capacities

def find_most_connected_ground_users(nodes_list, print_prog=False):
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

    if print_prog:
        print("Each UAV cover some GUs:")
        print(connection_counts)
        # print(most_common_node)

    if most_common_node == None:
        print("There is no frequently used UAV node")
        return [], None

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

def is_position_inside_block(position, blocks):
    x, y, z = position
    for block in blocks:
        bx, by, bz = block["bottomCorner"]
        sx, sy = block["size"]
        h = block['height']
        if (bx <= x <= bx + sx) and (by <= y <= by + sy) and (bz <= z <= bz + h):
            return True
    return False

import matplotlib.pyplot as plt

def plot_gu_capacities(capacities_tracks):
    """
    Plot the maximum capacity for each ground user over time, assuming each capacity value is a tuple
    with the capacity and the UAV ID that provides it.
    """
    gu_capacities = {}

    for capacities_dict in capacities_tracks:
        for gu, capacity_tuple in capacities_dict.items():
            # gu_id = id(gu)
            gu_id = gu.node_number
            max_capacity = capacity_tuple[0]  # Extract the max capacity from the tuple
            gu_capacities.setdefault(gu_id, []).append(max_capacity)

    plt.figure(figsize=(12, 8))
    for gu_id, capacities in gu_capacities.items():
        plt.plot(capacities, label=f'GU {gu_id}')

    plt.title('Max Capacity for each GU over time')
    plt.xlabel('Time/Scenario')
    plt.ylabel('Max Capacity')
    plt.legend()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_gu_uav_statistics(capacities_tracks):
    """
    Plot the minimum, mean, and maximum number of connections for GUs and the number of GUs
    managed by each UAV over time.

    Parameters:
    capacities_tracks: A list of dictionaries representing the capacity tracks over time,
                       where each dictionary maps GUs to a tuple of (capacity, UAV ID).
    """
    min_capacities = []
    mean_capacities = []
    max_capacities = []
    uav_responsibilities = {}  # UAV ID mapped to a list of GU counts over time

    for capacities_dict in capacities_tracks:
        capacities = [capacity_tuple[0] for capacity_tuple in capacities_dict.values()]
        min_capacities.append(min(capacities))
        mean_capacities.append(np.mean(capacities))
        max_capacities.append(max(capacities))

        # Count GUs per UAV
        for gu, (_, uav_id) in capacities_dict.items():

            # print("Current GU number:")
            # print(gu.node_number)
            # print("Related UAV number:")
            # print(uav_id)

            if uav_id not in uav_responsibilities:
                uav_responsibilities[uav_id] = [0] * len(capacities_tracks)
                # print("New uav is added:"+str(uav_id))
            uav_responsibilities[uav_id][capacities_tracks.index(capacities_dict)] += 1
            
            print(capacities_dict)
            print("Add to moment: "+str(capacities_tracks.index(capacities_dict)))

    time = list(range(len(capacities_tracks)))

    # Plotting the line chart for min, mean, and max capacities
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.plot(time, min_capacities, label='Min Capacity', marker='o')
    plt.plot(time, mean_capacities, label='Mean Capacity', marker='o')
    plt.plot(time, max_capacities, label='Max Capacity', marker='o')
    plt.title('Min, Mean, and Max GU Capacities over Time')
    plt.xlabel('Time/Scenario')
    plt.ylabel('Capacity')
    plt.legend()

    # Plotting the bar chart for UAV responsibilities
    plt.subplot(1, 2, 2)
    bottom = np.zeros(len(capacities_tracks))  # Starting point for each bar stack
    
    print("TEST")
    print(bottom)
    print(uav_responsibilities)


    for uav_id, counts in uav_responsibilities.items():
        plt.bar(time, counts, bottom=bottom, label=f'UAV {uav_id}')
        bottom += counts  # Increment the starting point for the next stack

    plt.title('GU Responsibilities per UAV over Time')
    plt.xlabel('Time/Scenario')
    plt.ylabel('Number of GUs')
    plt.legend()

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_gu_summary_and_uav_load(capacities_tracks):
    """
    Plot the summary statistics (min, mean, max) of GU capacities and the load on each UAV over time.

    Parameters:
    capacities_tracks: List of dictionaries, each mapping ground users to a tuple of capacity and UAV ID.
    """
    min_capacities = []
    mean_capacities = []
    max_capacities = []
    uav_loads_per_time = []

    # Process capacity data to compute summary statistics
    for capacities_dict in capacities_tracks:
        capacities = [cap[0] for cap in capacities_dict.values()]
        min_capacities.append(np.min(capacities))
        mean_capacities.append(np.mean(capacities))
        max_capacities.append(np.max(capacities))

        # Calculate the load on each UAV
        uav_loads = {}
        for cap in capacities_dict.values():
            uav_id = cap[1]
            uav_loads[uav_id] = uav_loads.get(uav_id, 0) + 1

        uav_loads_per_time.append(uav_loads)

    # Plotting the summary statistics of GU capacities
    time_points = range(len(capacities_tracks))
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(time_points, min_capacities, label='Min Capacity', marker='o')
    plt.plot(time_points, mean_capacities, label='Mean Capacity', marker='o')
    plt.plot(time_points, max_capacities, label='Max Capacity', marker='o')
    plt.title('Min, Mean, and Max GU Capacities Over Time')
    plt.xlabel('Time/Scenario')
    plt.ylabel('Capacity')
    plt.legend()

    # Plotting the load on each UAV as a bar chart
    plt.subplot(1, 2, 2)
    total_gus = sum([len(capacities_dict) for capacities_dict in capacities_tracks]) / len(capacities_tracks)
    bottom = np.zeros(len(time_points))

    # for uav_id in sorted({uav for loads in uav_loads_per_time for uav in loads}):
    #     loads = [uav_loads.get(uav_id, 0) for uav_loads in uav_loads_per_time]
    #     plt.bar(time_points, loads, bottom=bottom, label=f'UAV {uav_id}')
    #     bottom += loads
  
    for uav_id in {uav for loads in uav_loads_per_time for uav in loads}:
        loads = [uav_loads.get(uav_id, 0) for uav_loads in uav_loads_per_time]
        plt.bar(time_points, loads, bottom=bottom, label=f'UAV {uav_id}')
        bottom += loads

    plt.title('Load on UAVs Over Time')
    plt.xlabel('Time/Scenario')
    plt.ylabel('Number of GUs')
    plt.ylim(0, total_gus)
    plt.legend()

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_combined_gu_capacity_and_uav_load_separate(capacities_tracks):
    """
    Plot the summary statistics (min, mean, max) of GU capacities and the load on each UAV over time
    in a combined chart with shared x-axis and dual y-axis.

    Parameters:
    capacities_tracks: List of dictionaries, each mapping ground users to a tuple of capacity and UAV ID.
    """
    min_capacities = []
    mean_capacities = []
    max_capacities = []
    uav_loads_per_time = []

    # Process capacity data to compute summary statistics
    for capacities_dict in capacities_tracks:
        capacities = [cap[0] for cap in capacities_dict.values()]
        min_capacities.append(np.min(capacities))
        mean_capacities.append(np.mean(capacities))
        max_capacities.append(np.max(capacities))

        # Calculate the load on each UAV
        uav_loads = {}
        for cap in capacities_dict.values():
            uav_id = cap[1]
            uav_loads[uav_id] = uav_loads.get(uav_id, 0) + 1

        uav_loads_per_time.append(uav_loads)

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharex=True)

    # Plot the GU capacities on the first subplot (left)
    time_points = range(len(capacities_tracks))
    ax1.set_xlabel('Time/Scenario')
    ax1.set_ylabel('Capacity')
    ax1.plot(time_points, min_capacities, label='Min Capacity', marker='o', color='tab:blue')
    ax1.plot(time_points, mean_capacities, label='Mean Capacity', marker='o', color='tab:green')
    ax1.plot(time_points, max_capacities, label='Max Capacity', marker='o', color='tab:red')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')
    ax1.set_title('GU Capacities Over Time')

    # Plot the UAV loads on the second subplot (right)
    ax2.set_xlabel('Time/Scenario')
    ax2.set_ylabel('Number of GUs')
    total_gus = sum([len(capacities_dict) for capacities_dict in capacities_tracks]) / len(capacities_tracks)
    bottom = np.zeros(len(time_points))

    for uav_id in {uav for loads in uav_loads_per_time for uav in loads}:
        loads = [uav_loads.get(uav_id, 0) for uav_loads in uav_loads_per_time]
        ax2.bar(time_points, loads, label=f'UAV {uav_id}', alpha=0.5, edgecolor='black', bottom=bottom)
        bottom += loads

    ax2.tick_params(axis='y')
    ax2.set_ylim(0, total_gus)
    ax2.legend(loc='upper left')
    ax2.set_title('UAV Load Over Time')

    # Add a main title for the entire figure
    plt.suptitle('GU Capacities and UAV Load Over Time')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_combined_gu_capacity_and_uav_load(capacities_tracks):
    """
    Plot the summary statistics (min, mean, max) of GU capacities and the load on each UAV over time
    in a combined chart with shared x-axis and dual y-axis.

    Parameters:
    capacities_tracks: List of dictionaries, each mapping ground users to a tuple of capacity and UAV ID.
    """
    min_capacities = []
    mean_capacities = []
    max_capacities = []
    uav_loads_per_time = []

    # Process capacity data to compute summary statistics
    for capacities_dict in capacities_tracks:
        capacities = [cap[0] for cap in capacities_dict.values()]
        min_capacities.append(np.min(capacities))
        mean_capacities.append(np.mean(capacities))
        max_capacities.append(np.max(capacities))

        # Calculate the load on each UAV
        uav_loads = {}
        for cap in capacities_dict.values():
            uav_id = cap[1]
            uav_loads[uav_id] = uav_loads.get(uav_id, 0) + 1

        uav_loads_per_time.append(uav_loads)

    # Create the figure and the first subplot with shared x-axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot the GU capacities on the primary y-axis
    time_points = range(len(capacities_tracks))
    ax1.set_xlabel('Time/Scenario')
    ax1.set_ylabel('Capacity')
    ax1.plot(time_points, min_capacities, label='Min Capacity', marker='o', color='tab:blue')
    ax1.plot(time_points, mean_capacities, label='Mean Capacity', marker='o', color='tab:green')
    ax1.plot(time_points, max_capacities, label='Max Capacity', marker='o', color='tab:red')
    ax1.tick_params(axis='y')

    # Create the second y-axis for the UAV load
    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of GUs')
    total_gus = sum([len(capacities_dict) for capacities_dict in capacities_tracks]) / len(capacities_tracks)
    bottom = np.zeros(len(time_points))

    # Plot the UAV loads on the secondary y-axis
    for uav_id in {uav for loads in uav_loads_per_time for uav in loads}:
        loads = [uav_loads.get(uav_id, 0) for uav_loads in uav_loads_per_time]
        ax2.bar(time_points, loads, label=f'UAV {uav_id}', alpha=0.5, edgecolor='black', bottom=bottom)
        bottom += loads

    ax2.tick_params(axis='y')
    ax2.set_ylim(0, total_gus)

    # Add legends and title
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('GU Capacities and UAV Load Over Time')

    # plt.subplots_adjust(top=0.95)

    plt.show()
