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
    # 创建颜色映射，每个时间单位一个颜色
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    
    # 初始化图表
    fig, ax = plt.subplots()
    ax.set_xlim(0, xLength)
    ax.set_ylim(0, yLength)
    
    # 绘制blocks
    for block in blocks:
        bx, by, _ = block['bottomCorner']
        bw, bh = block['size']
        block_rect = patches.Rectangle((bx, by), bw, bh, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(block_rect)

    # 为每个时间单位绘制所有用户的移动
    for time_step in range(n):
        # 移动用户
        move_ground_users(ground_users, blocks, xLength, yLength, max_movement_distance)
        # 绘制移动路径
        for gu in ground_users:
            # 如果是第一个时间步骤，记录起始位置
            if time_step == 0:
                gu.start_pos = gu.position
            # 否则，使用上一个时间步骤的终点作为起点
            else:
                gu.start_pos = gu.end_pos

            # 记录新位置
            gu.end_pos = gu.position
            
            # 绘制从起始位置到新位置的箭头
            ax.arrow(gu.start_pos[0], gu.start_pos[1], gu.end_pos[0] - gu.start_pos[0], gu.end_pos[1] - gu.start_pos[1],
                     head_width=0.5, head_length=1, fc=colors[time_step], ec=colors[time_step])

        # 添加时间步骤到图例
        ax.plot([], [], color=colors[time_step], label=f'Time Step: {time_step + 1}')
    
    # 添加图例
    ax.legend(loc='upper left')
    
    # 显示最终的可视化结果
    plt.show()



# Load scene data from JSON file
with open('scene_data_simple.json', 'r') as file:
    scene_data = json.load(file)

blocks = scene_data['blocks']
scene = scene_data['scenario']

ground_users = generate_users(10, blocks, scene['xLength'], scene['yLength'])

# print_nodes(ground_users, True)

max_movement_distance = 15

# 假定其他变量已经按照你的代码加载和定义

simulate_and_visualize_movements(5, ground_users, blocks, scene['xLength'], scene['yLength'], max_movement_distance)