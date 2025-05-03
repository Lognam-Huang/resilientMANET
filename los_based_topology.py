import math
from classes.BackhaulPaths import BackhaulPaths
from functions.path_is_blocked import path_is_blocked
from functions.calculate_data_rate import calculate_data_rate

from connectivity_finding import get_backhaul_connection, reward

def find_los_backhaul_topology(GU_nodes, UAV_nodes, BS_nodes, scene_info, reward_hyper):
    """
    Find the backhaul topology based on LOS connections between nodes.

    Parameters:
    GU_nodes: List of ground user nodes.
    UAV_nodes: List of UAV nodes.
    BS_nodes: List of base station nodes.
    scene_info: Dictionary containing scenario information (e.g., obstacles, UAV properties).
    reward_hyper: Dictionary containing reward hyperparameters.

    Returns:
    Same as find_best_backhaul_topology:
    - best_state: Binary string representing the best topology.
    - max_reward: Maximum reward achieved.
    - best_resilience_score: Best resilience score achieved.
    - reward_track: List of rewards for each evaluated topology.
    - RS_track: List of resilience scores for each evaluated topology.
    - best_reward_track: List of the best rewards found during the process.
    - best_RS_track: List of the best resilience scores found during the process.
    - best_backhaul_connection: BackhaulPaths object representing the best topology.
    """
    # Initialize variables
    best_state = ""
    max_reward = -float('inf')
    best_resilience_score = -float('inf')
    reward_track = []
    RS_track = []
    best_reward_track = []
    best_RS_track = []
    best_backhaul_connection = None

    # Number of nodes
    num_uavs = len(UAV_nodes)
    num_bss = len(BS_nodes)
    total_nodes = num_uavs + num_bss

    # Generate the binary state based on LOS connections
    state = []
    for i in range(total_nodes):
        for j in range(i + 1, total_nodes):
            if i < num_uavs and j < num_uavs:
                # UAV-UAV connection
                los = not path_is_blocked(scene_info['blocks'], UAV_nodes[i], UAV_nodes[j])
            elif i < num_uavs and j >= num_uavs:
                # UAV-BS connection
                los = not path_is_blocked(scene_info['blocks'], UAV_nodes[i], BS_nodes[j - num_uavs])
            elif i >= num_uavs and j >= num_uavs:
                # BS-BS connection
                los = not path_is_blocked(scene_info['blocks'], BS_nodes[i - num_uavs], BS_nodes[j - num_uavs])
            else:
                los = False

            # Append '1' if LOS exists, otherwise '0'
            state.append('1' if los else '0')

    # Convert the state list to a binary string
    state = ''.join(state)

    # Calculate the backhaul connection for the generated state
    backhaul_connection = get_backhaul_connection(state, UAV_nodes, BS_nodes, scene_info)

    # Calculate the reward and resilience score for the state
    reward_score, resilience_score, backhaul_connection = reward(state, scene_info, GU_nodes, UAV_nodes, BS_nodes, reward_hyper)

    # Update the best state if the current reward is higher
    if reward_score > max_reward:
        best_state = state
        max_reward = reward_score
        best_resilience_score = resilience_score
        best_backhaul_connection = backhaul_connection

    # Track the reward and resilience score
    reward_track.append(reward_score)
    RS_track.append(resilience_score)
    best_reward_track.append(max_reward)
    best_RS_track.append(best_resilience_score)

    # Return the results
    return best_state, max_reward, best_resilience_score, reward_track, RS_track, best_reward_track, best_RS_track, best_backhaul_connection