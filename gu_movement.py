import json
from functions.generate_users import generate_users
from functions.print_nodes import *

import random
import math
import numpy as np

def move_ground_users(ground_users, blocks, xLength, yLength, max_movement_distance):
    for gu in ground_users:
        moved = False
        while not moved:
            # 生成随机移动方向和距离
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, max_movement_distance)
            dx = distance * math.cos(angle)
            dy = distance * math.sin(angle)

            # 获取当前位置并计算新位置
            current_position = gu.position
            new_x = current_position[0] + dx
            new_y = current_position[1] + dy

            # 检查新位置是否在场景内
            if 0 <= new_x <= xLength and 0 <= new_y <= yLength:
                collision = False
                # 检查新位置是否与任何块碰撞
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



if __name__ == "__main__":

    # Load scene data from JSON file
    with open('scene_data_simple.json', 'r') as file:
        scene_data = json.load(file)

    blocks = scene_data['blocks']
    scene = scene_data['scenario']

    ground_users = generate_users(10, blocks, scene['xLength'], scene['yLength'])

    # print_nodes(ground_users, True)

    max_movement_distance = 3

    simulate_and_visualize_movements(10, ground_users, blocks, scene['xLength'], scene['yLength'], max_movement_distance)