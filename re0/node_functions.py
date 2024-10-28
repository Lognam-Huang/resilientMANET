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

def move_ground_users(ground_users, blocks, xLength, yLength, max_movement_distance):
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
                    if bx <= new_x <= bx + bw and by <= new_y <= by + bh:
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

def get_gu_to_uav_connections(ground_users, UAV_nodes, UAVInfo, blocks, backhaul_connection):
    gu_to_uav = {}

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
    
    if backhaul_connection:
        for uav_index, uav in enumerate(UAV_nodes):
            if uav_index in backhaul_connection.allPaths:
                max_backhaul_dr = -1
                best_backhaul_path = None
                for connection in backhaul_connection.allPaths[uav_index]:
                    path = connection['path']
                    dr = connection['DR']

                    if dr > max_backhaul_dr:
                        max_backhaul_dr = dr
                        best_backhaul_path = path[1]
                    
                    for relay_node in path:
                        if relay_node == uav_index: continue
                        if relay_node < len(UAV_nodes) and not relay_node in UAV_nodes[uav_index].connected_nodes:
                            UAV_nodes[uav_index].add_connection(relay_node)
                
                if best_backhaul_path is not None:
                    UAV_nodes[uav_index].set_DR(max_backhaul_dr)

    return gu_to_uav

def move_gu_and_update_connections(ground_users, blocks, x_length, y_length, max_movement_distance, UAV_nodes, UAVInfo, backhaul_connection):
    move_ground_users(ground_users, blocks, x_length, y_length, max_movement_distance)

    gu_to_uav_connections = get_gu_to_uav_connections(ground_users, UAV_nodes, UAVInfo, blocks, backhaul_connection)
    
    return gu_to_uav_connections