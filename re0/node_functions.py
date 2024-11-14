import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes.Nodes import Nodes

def generate_nodes(node_number, nodeType, default_height = 100):
    node_list = []
    node_index = 0

    for i in range(node_number):
        if (nodeType == 0):    
            node_list.append(Nodes((0,0,0), "GU", 0, node_index))
        elif (nodeType == 1):
            node_list.append(Nodes((0,0,default_height), "UAV", 0, node_index))
        elif (nodeType ==2):
            node_list.append(Nodes((0,0,default_height), "ABS", 0, node_index))
        else:
            TypeError()
            break
        node_index += 1
    
    return node_list

def print_node(all_nodes, node_number = -1, onlyPosition = False):
    for node in all_nodes:
        if node_number == -1:
            if onlyPosition == True:
                print(node.position)
            else:
                print(node)
        else:
            if node.node_number == node_number:
                if onlyPosition == True:
                    print(node.position)
                else:
                    print(node)

def print_node_number(node):
    print(node.node_number)

def get_nodes_position(all_nodes):
    positions = []
    for node in all_nodes:
        positions.append(node.position)
    return positions

import json
import random
import math
import numpy as np

def move_ground_users(ground_users, blocks, xLength, yLength, max_movement_distance, soft_margin=-1):
    if soft_margin == -1: soft_margin = min(xLength, yLength)/100
    for gu in ground_users:
        moved = False
        while not moved:
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, max_movement_distance)
            dx = distance * math.cos(angle)
            dy = distance * math.sin(angle)

            current_position = gu.position
            new_x = current_position[0] + dx
            new_y = current_position[1] + dy

            if 0 <= new_x <= xLength and 0 <= new_y <= yLength:
                collision = False
                for block in blocks:
                    bx, by, _ = block['bottomCorner']
                    bw, bh = block['size']
                    if (bx - soft_margin <= new_x <= bx + bw + soft_margin and 
                        by - soft_margin <= new_y <= by + bh + soft_margin):
                       
                        collision = True
                        break
                
                if not collision:
                    gu.set_position((new_x, new_y, current_position[2]))
                    moved = True

import matplotlib.pyplot as plt
import matplotlib.patches as patches
def simulate_and_visualize_movements(n, ground_users, blocks, xLength, yLength, max_movement_distance):
    # Create a color map with a unique color for each time step
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    
    # Initialize the plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, xLength)
    ax.set_ylim(0, yLength)
    
    # Draw the blocks on the plot
    for block in blocks:
        bx, by, _ = block['bottomCorner']
        bw, bh = block['size']
        block_rect = patches.Rectangle((bx, by), bw, bh, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(block_rect)

    # For each time step, draw the movement of all users
    for time_step in range(n):
        # Initialize the starting positions for the first time step
        if time_step == 0:
            for gu in ground_users:
                gu.start_pos = gu.position
                
        # Move the users
        move_ground_users(ground_users, blocks, xLength, yLength, max_movement_distance)
        
        # Draw the movement paths
        for gu in ground_users:
            # Update the start position for subsequent time steps
            if time_step != 0:
                gu.start_pos = gu.end_pos

            # Record the new position
            gu.end_pos = gu.position
            
            # Draw an arrow from the start to the new position
            ax.arrow(gu.start_pos[0], gu.start_pos[1], gu.end_pos[0] - gu.start_pos[0], gu.end_pos[1] - gu.start_pos[1],
                     head_width=0.5, head_length=1, fc=colors[time_step], ec=colors[time_step])

        # Add a legend entry for this time step
        ax.plot([], [], color=colors[time_step], label=f'Time Step: {time_step + 1}')
    
    # Add the legend to the plot
    ax.legend(loc='upper left')
    
    # Display the final visualization
    plt.show()

from functions.path_is_blocked import path_is_blocked
from functions.calculate_data_rate import calculate_data_rate

def get_gu_to_uav_connections(ground_users, UAV_nodes, UAVInfo, blocks, backhaul_connection = None):
    gu_to_uav = {}
    gu_to_bs_capacity = {}

    if backhaul_connection:
        for uav_index, uav in enumerate(UAV_nodes):
            if uav_index in backhaul_connection.allPaths:
                max_backhaul_dr = -1
                for connection in backhaul_connection.allPaths[uav_index]:
                    path = connection['path']
                    dr = connection['DR']

                    if dr > max_backhaul_dr:
                        max_backhaul_dr = dr
                    
                    for relay_node in path:
                        if relay_node == uav_index: continue
                        if relay_node < len(UAV_nodes) and not relay_node in UAV_nodes[uav_index].connected_nodes:
                            UAV_nodes[uav_index].add_connection(relay_node)
                
                if max_backhaul_dr > 0:
                    UAV_nodes[uav_index].set_DR(max_backhaul_dr)

    for gu_index, user in enumerate(ground_users):
        max_dr = -1
        best_uav = None
        for uav_index, uav in enumerate(UAV_nodes):
            blocked = path_is_blocked(blocks, uav, user)
            dr = calculate_data_rate(UAVInfo, uav.position, user.position, blocked)
            if dr > max_dr:
                max_dr = dr
                best_uav = uav_index
        gu_to_uav[gu_index] = [best_uav]

        ground_users[gu_index].set_connection(best_uav)
        ground_users[gu_index].set_DR(max_dr)

        gu_to_bs_capacity[gu_index] = min(max_dr, UAV_nodes[user.connected_nodes[0]].data_rate[0])
    
    return gu_to_uav, gu_to_bs_capacity

def move_gu_and_update_connections(ground_users, blocks, x_length, y_length, max_movement_distance, UAV_nodes, UAVInfo, backhaul_connection):
    move_ground_users(ground_users, blocks, x_length, y_length, max_movement_distance)

    gu_to_uav_connections, gu_to_bs_capacity = get_gu_to_uav_connections(ground_users, UAV_nodes, UAVInfo, blocks, backhaul_connection)
    
    return gu_to_uav_connections, gu_to_bs_capacity

def add_gu_to_simulation(ground_user, number_of_added_gu):
    for i in range(number_of_added_gu):
        ground_user.append(Nodes((0,0,0), "GU", 0, len(ground_user)))
    return ground_user

def add_or_remove_gu(ground_user):
    while True:
        # Randomly select an operation: 2 means add 2 GUs, 1 means add 1 GU, 0 means no change,
        # -1 means remove 1 GU, and -2 means remove 2 GUs
        operation = random.choice([2, 1, 0, -1, -2])
        
        # Print the selected operation and perform the corresponding addition or removal
        if operation > 0:
            print(f"Selected to add {operation} GU(s)")
            for _ in range(operation):  # Add the specified number of GUs
                ground_user.append(Nodes((0,0,0), "GU", 0, len(ground_user)))
            break  # Operation is valid, exit the loop
        elif operation == 0:
            print("Selected to make no changes")
            break  # Operation is valid, exit the loop
        elif operation < 0:
            # Check if there are enough GUs to remove
            if len(ground_user) >= abs(operation):
                print(f"Selected to remove {abs(operation)} GU(s)")
                for _ in range(abs(operation)):  # Remove the specified number of GUs
                    ground_user.pop()
                break  # Operation is valid, exit the loop
            else:
                print("Not enough GUs to remove; reselecting operation.")
                # Continue the loop to select a new operation
    
    return ground_user

def set_baseline_backhaul_for_mid_scene(baseline_UAV_nodes, baseline_UAV_positions, baseline_UAV_connections, baseline_BS_nodes, baseStation, baseline_BS_connections):
    for uav_index, uav_nodes in enumerate(baseline_UAV_nodes):
        uav_nodes.set_position(baseline_UAV_positions[uav_index]) if baseline_UAV_positions and baseline_UAV_positions[uav_index] else None
        uav_nodes.set_connection(baseline_UAV_connections[uav_index]) if baseline_UAV_connections and baseline_UAV_connections[uav_index] else None
    
    for i in range(len(baseline_BS_nodes)):
        baseline_BS_nodes[i].set_position((baseStation[i]['bottomCorner'][0], baseStation[i]['bottomCorner'][1], baseStation[i]['height'][0]))
        baseline_BS_nodes[i].set_connection(baseline_BS_connections[i])
