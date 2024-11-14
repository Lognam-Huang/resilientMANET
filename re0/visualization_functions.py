import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def scene_visualization(ground_users = None, UAV_nodes = None, air_base_station = None, scene_info = None,line_alpha=0, show_axes_labels=True):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # print(scene_info['xLength'])
    blocks = scene_info['blocks']
    # print(scene_info['scenario'])
    
    ax.set_xlim([0, scene_info['scenario']['xLength']])
    ax.set_ylim([0, scene_info['scenario']['yLength']])

    max_block_height = 0
    
    max_block_height = max((block['height'] for block in blocks), default=0) if blocks else 0
    max_uav_height = max((UAV.position[2] for UAV in UAV_nodes), default=0) if UAV_nodes else 0
    max_abs_height = max((ABS.position[2] for ABS in air_base_station), default=0) if air_base_station else 0
    
    max_height = max(max_block_height, max_uav_height, max_abs_height) * 1.2  # 取最大值并乘以120%
    ax.set_zlim([0, max_height])
    
    
    if show_axes_labels:
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
    else:
        # ax.set_xticks(ax.get_xticks())
        # ax.set_yticks(ax.get_yticks())
        # ax.set_zticks(ax.get_zticks())
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        # ax.set_xlabel('')
        # ax.set_ylabel('')
        # ax.set_zlabel('')

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # set box size of each UAV_nodes
    # node_size = 2
    node_size = min(scene_info['scenario']['xLength'], scene_info['scenario']['yLength'])/40
    
    if blocks:    
        for block in blocks:    
            # print(block)
            # print(block['bottomCorner'])            
            x, y, z = block['bottomCorner']
            dx, dy = block['size']
            dz = block['height']
            color = (1, 1, 1, 0.5)
            # label = block['label']
            
            ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=color)

            # 添加标记文字在块的中心
            # ax.text(x + dx/2, y + dy/2, dz/2, label, color='black', ha='center', va='center') 
        
    if ground_users:            
        for user in ground_users:        
            # print(user.position[0])
        
            x, y = user.position[0], user.position[1]
            dx, dy, dz= node_size, node_size, node_size
            color = (0, 0, 1, 0.5)
            # label = block['label']

            # print("GU")
            # print(x, y, 0, dx, dy, dz)
            
            ax.bar3d(x, y, 0, dx, dy, dz, shade=True, color=color)

    if UAV_nodes:                 
        for UAV in UAV_nodes:        
            x, y, z = UAV.position[0], UAV.position[1], UAV.position[2]
            dx, dy, dz= node_size, node_size, node_size
            color = (0, 1, 0, 0.5)
            # label = block['label']

            # print("UAV")
            # print(x, y, z, dx, dy, dz)
            
            ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=color)


    if air_base_station:
        for ABS in air_base_station:        
            x, y, z = ABS.position[0], ABS.position[1], ABS.position[2]
            dx, dy, dz= node_size, node_size, node_size
            color = (1, 0, 0, 0.5)
            # label = block['label']
            
            ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=color)
    
    # # 可视化heatmap
    # if heatmap is not None:
    #     max_users = np.max(heatmap)  # 获取最大的ground_user数
    #     for x in range(heatmap.shape[0]):
    #         for y in range(heatmap.shape[1]):
    #             for z in range(heatmap.shape[2]):
    #                 value = heatmap[x, y, z]
    #                 if value > 0:  # 如果该点的值大于0
    #                     alpha = (value / max_users) * 0.02  # 根据ground_user的数量调整透明度
    #                     ax.bar3d(x, y, z+min_height, 1, 1, 1, shade=False, color=(0, 1, 0, alpha))

    # if connection_GU_UAV and ground_users and UAV_nodes:
    #     for gu_index, uav_index in connection_GU_UAV:
    #         gu = ground_users[gu_index]
    #         uav = UAV_nodes[uav_index]
    #         gu_pos = np.array([gu.position[0], gu.position[1], 0])
    #         uav_pos = np.array([uav.position[0], uav.position[1], uav.position[2]])
    #         ax.plot([gu_pos[0], uav_pos[0]], [gu_pos[1], uav_pos[1]], [gu_pos[2], uav_pos[2]], color='k')

    # if connection_GU_UAV:
    #     for start, end in connection_GU_UAV:

    #         print(start)
    #         print(end)

    #         start_pos = get_position_by_index(start, ground_users, UAV_nodes, air_base_station)
    #         end_pos = get_position_by_index(end, ground_users, UAV_nodes, air_base_station)
    #         ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], color='k')


    line_color = (0.5,0,0)
    
    from node_functions import print_node
    # print_node(UAV_nodes)
    
    for gu in ground_users:
        gu_x, gu_y, gu_z = gu.position[0], gu.position[1], gu.position[2]
        uav_index =  gu.connected_nodes[0]
        uav_x, uav_y, uav_z = UAV_nodes[uav_index].position[0], UAV_nodes[uav_index].position[1], UAV_nodes[uav_index].position[2]

        ax.plot([gu_x, uav_x], [gu_y, uav_y], [gu_z, uav_z], color=(0.5,0,0), alpha=line_alpha)
    
    for uav in UAV_nodes:
        start_uav_x, start_uav_y, start_uav_z = uav.position[0], uav.position[1], uav.position[2]
        for target_uav_index in uav.connected_nodes:
            # print(target_uav_index)
            target_uav_x, target_uav_y, target_uav_z = UAV_nodes[target_uav_index].position[0], UAV_nodes[target_uav_index].position[1], UAV_nodes[target_uav_index].position[2]

            ax.plot([start_uav_x, target_uav_x], [start_uav_y, target_uav_y], [start_uav_z, target_uav_z], color=(0.5,0,0), alpha=line_alpha)
    
    for bs in air_base_station:
        bs_x, bs_y, bs_z = bs.position[0], bs.position[1], bs.position[2]
        for target_uav_index in bs.connected_nodes:
            target_uav_x, target_uav_y, target_uav_z = UAV_nodes[target_uav_index].position[0], UAV_nodes[target_uav_index].position[1], UAV_nodes[target_uav_index].position[2]

            ax.plot([bs_x, target_uav_x], [bs_y, target_uav_y], [bs_z, target_uav_z], color=(0.5,0,0), alpha=line_alpha)
        
    # GU-UAV连接线
    # if ground_users and UAV_nodes:
    #     for user in ground_users:
    #         # print(user.position)
    #         gu_x, gu_y, gu_z = user.position[0], user.position[1], 0  # Ground users 的 z 坐标为 0
    #         for UAV in UAV_nodes:  # 假设所有 GU 都与 UAV 相连
    #             print(UAV.position)
    #             uav_x, uav_y, uav_z = UAV.position[0], UAV.position[1], UAV.position[2]
    #             ax.plot([gu_x, uav_x], [gu_y, uav_y], [gu_z, uav_z], color=line_color, alpha=line_alpha)

    # UAV-UAV连接线
    # if UAV_nodes:
    #     for i, UAV1 in enumerate(UAV_nodes):
    #         for j, UAV2 in enumerate(UAV_nodes):
    #             if i < j:  # 避免重复连接
    #                 uav1_x, uav1_y, uav1_z = UAV1.position[0], UAV1.position[1], UAV1.position[2]
    #                 uav2_x, uav2_y, uav2_z = UAV2.position[0], UAV2.position[1], UAV2.position[2]
    #                 ax.plot([uav1_x, uav2_x], [uav1_y, uav2_y], [uav1_z, uav2_z], color=line_color, alpha=line_alpha)

    # # UAV-BS连接线
    # if UAV_nodes and air_base_station:
    #     for UAV in UAV_nodes:
    #         uav_x, uav_y, uav_z = UAV.position[0], UAV.position[1], UAV.position[2]
    #         for ABS in air_base_station:
    #             abs_x, abs_y, abs_z = ABS.position[0], ABS.position[1], ABS.position[2]
    #             ax.plot([uav_x, abs_x], [uav_y, abs_y], [uav_z, abs_z], color=line_color, alpha=line_alpha)

    # if connection_GU_UAV:
    #     for gu, path in connection_GU_UAV.items():
    #         if path:
    #             start = gu
    #             end = path[0]+len(ground_users)
    #             start_pos = get_position_by_index(start, ground_users, UAV_nodes, air_base_station)
    #             end_pos = get_position_by_index(end, ground_users, UAV_nodes, air_base_station)

    #             ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], color=line_color, alpha=line_alpha)
    
    # if connection_UAV_BS:
    #     for uav, path in connection_UAV_BS.items():
    #         for i in range(len(path) - 1):
    #             start = path[i]+len(ground_users)
    #             end = path[i + 1]+len(ground_users)
    #             start_pos = get_position_by_index(start, ground_users, UAV_nodes, air_base_station)
    #             end_pos = get_position_by_index(end, ground_users, UAV_nodes, air_base_station)

    #             ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], color=line_color, alpha=line_alpha)

    plt.show()

def visualize_all_gu_capacity(gu_capacity_TD):
    """
    Visualizes the data rate of each ground user (GU) over time.
    
    Parameters:
    gu_capacity_TD (list of lists): A list where each inner list contains data rates of GUs at a specific time step.
    
    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    
    # Plot each ground user's capacity over time
    for gu_index in range(len(gu_capacity_TD[0])):  # assuming all time steps have the same number of GUs
        gu_data = [time_step[gu_index] for time_step in gu_capacity_TD]
        plt.plot(gu_data, label=f'GU {gu_index + 1}')
    
    plt.xlabel("Time Step")
    plt.ylabel("Data Rate (bps)")
    plt.title("Ground User Capacity Over Time")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()


def visualize_all_min_gu_capacity(min_gu_capacity_TD):
    plt.figure(figsize=(10, 6))
    
    gu_indices = list(range(1, len(min_gu_capacity_TD) + 1))  
    plt.plot(gu_indices, min_gu_capacity_TD, marker='o') 
    
    plt.xlabel("Time Step")
    plt.ylabel("Data Rate (bps)")
    plt.title("Minimum Ground User Capacity Over Time")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
   

def visualize_metrics(max_reward_TD, max_RS_TD, max_OL_TD=None):

    """
    可视化 max_reward_TD, max_RS_TD 和 max_OL_TD 在同一张图中
    :param max_reward_TD: List of max reward over time
    :param max_RS_TD: List of max resilience score over time
    :param max_OL_TD: List of max overload score over time
    """
    # 假设每个列表的数据长度相同，并且每个索引对应同一个时间点
    time_points = list(range(len(max_reward_TD)))

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 绘制每条曲线
    plt.plot(time_points, max_reward_TD, label='Max Reward TD', marker='o')
    plt.plot(time_points, max_RS_TD, label='Max RS TD', marker='s')
    # plt.plot(time_points, max_OL_TD, label='Max OL TD', marker='^')

    # 添加标签和标题
    plt.xlabel('Time Points')
    plt.ylabel('Scores')
    plt.title('Max Reward TD, Max RS TD Over Time')
    plt.legend()
    plt.grid(True)

    # 显示图表
    plt.show()

def get_position_by_index(index, ground_users, UAV_nodes, air_base_station):
    all_nodes = (ground_users or []) + (UAV_nodes or []) + (air_base_station or [])
    if 0 <= index < len(all_nodes):
        node = all_nodes[index]
        return node.position
    return [0, 0, 0]

def visualize_heatmap_slice(heatmap, target_height):
    """
    可视化特定高度的二维热图切片，包括 connection_score 和 gu_bottleneck。
    
    参数：
    - heatmap: 字典格式的热图数据，键为 (x, y, z)，值为 (connection_score, gu_bottleneck)。
    - target_height: 想要可视化的高度。
    - colormap: 颜色映射，默认 'hot'。
    """
    # 提取该高度的x, y以及相应的connection_score和gu_bottleneck
    x_vals, y_vals, connection_scores, gu_bottlenecks = [], [], [], []

    for (x, y, z), (connection_score, gu_bottleneck) in heatmap.items():
        if z == target_height:
            x_vals.append(x)
            y_vals.append(y)
            connection_scores.append(connection_score)
            gu_bottlenecks.append(gu_bottleneck)

    if not connection_scores or not gu_bottlenecks:
        print(f"No data available for height {target_height}.")
        return

    # 创建网格数据
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    x_unique = np.unique(x_vals)
    y_unique = np.unique(y_vals)
    x_grid, y_grid = np.meshgrid(x_unique, y_unique)

    # 重塑连接分数和瓶颈值为网格格式
    connection_score_grid = np.full(x_grid.shape, np.nan)
    gu_bottleneck_grid = np.full(x_grid.shape, np.nan)

    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
        x_index = np.where(x_unique == x)[0][0]
        y_index = np.where(y_unique == y)[0][0]
        connection_score_grid[y_index, x_index] = connection_scores[i]
        gu_bottleneck_grid[y_index, x_index] = gu_bottlenecks[i]

    # 可视化 connection_score 和 gu_bottleneck
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 可视化 connection_score
    c1 = axes[0].pcolormesh(x_grid, y_grid, connection_score_grid, cmap='hot', shading='auto')
    fig.colorbar(c1, ax=axes[0], label='Connection Score')
    axes[0].set_title(f'Connection Score at Height {target_height}')
    axes[0].set_xlabel("X Axis")
    axes[0].set_ylabel("Y Axis")
    
    # 可视化 gu_bottleneck
    c2 = axes[1].pcolormesh(x_grid, y_grid, gu_bottleneck_grid, cmap='viridis', shading='auto')
    fig.colorbar(c2, ax=axes[1], label='GU Bottleneck')
    axes[1].set_title(f'GU Bottleneck at Height {target_height}')
    axes[1].set_xlabel("X Axis")
    axes[1].set_ylabel("Y Axis")

    plt.tight_layout()
    plt.show()

def visualize_heatmap_slices(heatmap, target_heights, normalize=False):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12)) 
    colormaps = ['hot', 'viridis'] 
    
    for idx, target_height in enumerate(target_heights):
        x_vals, y_vals, connection_scores, gu_bottlenecks = [], [], [], []
        
        for (x, y, z), (connection_score, gu_bottleneck) in heatmap.items():
            if z == target_height:
                x_vals.append(x)
                y_vals.append(y)
                connection_scores.append(connection_score)
                gu_bottlenecks.append(gu_bottleneck)
        
        if not connection_scores or not gu_bottlenecks:
            print(f"No data available for height {target_height}.")
            continue

        # 如果需要归一化，使用min-max归一化
        if normalize:
            max_conn = max(connection_scores)
            min_conn = min(connection_scores)
            max_bottle = max(gu_bottlenecks)
            min_bottle = min(gu_bottlenecks)
            connection_scores = [(cs - min_conn) / (max_conn - min_conn) if max_conn != min_conn else 0 for cs in connection_scores]
            gu_bottlenecks = [(gb - min_bottle) / (max_bottle - min_bottle) if max_bottle != min_bottle else 0 for gb in gu_bottlenecks]

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        x_unique = np.unique(x_vals)
        y_unique = np.unique(y_vals)
        x_grid, y_grid = np.meshgrid(x_unique, y_unique)

        connection_score_grid = np.full(x_grid.shape, np.nan)
        gu_bottleneck_grid = np.full(x_grid.shape, np.nan)

        for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            x_index = np.where(x_unique == x)[0][0]
            y_index = np.where(y_unique == y)[0][0]
            connection_score_grid[y_index, x_index] = connection_scores[i]
            gu_bottleneck_grid[y_index, x_index] = gu_bottlenecks[i]

        c1 = axes[idx, 0].pcolormesh(x_grid, y_grid, connection_score_grid, cmap=colormaps[0], shading='auto')
        fig.colorbar(c1, ax=axes[idx, 0], label='Connection Score' + (' (Normalized)' if normalize else ''))
        axes[idx, 0].set_title(f'Connection Score at Height {target_height}')
        axes[idx, 0].set_xlabel("X Axis")
        axes[idx, 0].set_ylabel("Y Axis")
        
        c2 = axes[idx, 1].pcolormesh(x_grid, y_grid, gu_bottleneck_grid, cmap=colormaps[1], shading='auto')
        fig.colorbar(c2, ax=axes[idx, 1], label='GU Bottleneck' + (' (Normalized)' if normalize else ''))
        axes[idx, 1].set_title(f'GU Bottleneck at Height {target_height}')
        axes[idx, 1].set_xlabel("X Axis")
        axes[idx, 1].set_ylabel("Y Axis")

    plt.tight_layout()
    plt.show()


import matplotlib.patches as patches

from collections import defaultdict

def merge_clusters(clusters_records):
    """
    根据给定逻辑合并聚类，提取指定的较短或较长的 GU 列表作为最终类。

    参数：
    - clusters_records: 每次聚类的结果记录列表 [{0: [1, 2, 3], 1: [0, 4, 5]}, {0: [2, 3], 1: [1]}].

    返回:
    - classes: 最终的GU分组列表，按照给定规则从输入中提取。
    """
    classes = []

    # 遍历每个聚类记录
    for i, clusters in enumerate(clusters_records):
        lists = list(clusters.values())
        # 选择较短的列表作为类，若长度相同，选择第一个
        if len(lists[0]) < len(lists[1]):
            classes.append(lists[0])
        elif len(lists[0]) > len(lists[1]):
            classes.append(lists[1])
        else:
            classes.append(lists[0])

    # 将最后一个元素的较长列表记录为类，若长度相同，选择第二个
    final_lists = list(clusters_records[-1].values())
    if len(final_lists[0]) > len(final_lists[1]):
        classes.append(final_lists[0])
    elif len(final_lists[0]) < len(final_lists[1]):
        classes.append(final_lists[1])
    else:
        classes.append(final_lists[1])

    return classes

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from itertools import cycle

def visualize_hierarchical_clustering(ground_users, clusters_records, blocks, scene):
    """
    可视化层次聚类的最终GU分类。
    
    参数：
    - ground_users: 所有GU节点的列表，包含其位置属性 (x, y)。
    - clusters_records: 每次聚类的结果记录列表 [{0: [0, 1], 1: [2, 3, 4]}]。
    - blocks: 障碍物位置列表，每个元素是一个包含 "bottomCorner" 和 "size" 的字典。
    - scene: 场景信息，包含边界信息，用于设置绘图范围。
    """
    # # 合并聚类结果
    merged_clusters = merge_clusters(clusters_records)
    
    # 删除空聚类
    merged_clusters = [cluster for cluster in merged_clusters if cluster]

    other_gu_idx = []
    for gu_idx in range(len(ground_users)):
        found = False
        for clustered_gu_idxes in merged_clusters:
            if gu_idx in clustered_gu_idxes:
                found = True
                break
        if not found:
            other_gu_idx.append(gu_idx)
    merged_clusters.append(other_gu_idx) if len(other_gu_idx) > 0 else None

    print(merged_clusters)
    
    # 手动定义一组高度对比的颜色
    high_contrast_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#a55194', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    color_cycle = cycle(high_contrast_colors)
    colors = [next(color_cycle) for _ in range(len(merged_clusters))]

    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制障碍物
    for block in blocks:
        x, y, _ = block["bottomCorner"]
        width, height = block["size"]
        block_patch = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor='gray', alpha=0.5)
        ax.add_patch(block_patch)

    # 绘制合并后的每个类
    for cluster_idx, gu_indices in enumerate(merged_clusters):
        cluster_positions = [ground_users[index].position[:2] for index in gu_indices]
        x_vals, y_vals = zip(*cluster_positions)
        ax.scatter(x_vals, y_vals, color=colors[cluster_idx], label=f"Cluster {cluster_idx}", s=120, alpha=0.8, edgecolors='w')

    # 设置图形边界和标题
    ax.set_xlim(0, scene["xLength"])
    ax.set_ylim(0, scene["yLength"])
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_title("Final Hierarchical Clustering of Ground Users")
    ax.legend()
    plt.grid(True)
    plt.show()

def visualize_gu_by_connection(ground_users, blocks, scene):
    """
    可视化GU，每个GU的颜色根据其 .connected_nodes 的值确定，图例显示为“UAV 0, UAV 1...”格式。
    
    参数：
    - ground_users: GU节点列表，每个节点包含 position 属性 (x, y) 和 connected_nodes 列表。
    - blocks: 障碍物位置列表，每个元素是一个包含 "bottomCorner" 和 "size" 的字典。
    - scene: 场景信息，包含边界信息，用于设置绘图范围。
    """
    # 定义一组颜色用于表示不同连接类型
    connection_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_cycle = cycle(connection_colors)
    
    # 获取每种连接类型的颜色映射，按照连接类型从小到大排序
    unique_connections = sorted({gu.connected_nodes[0] for gu in ground_users})
    connection_types = {connection: next(color_cycle) for connection in unique_connections}
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制障碍物
    for block in blocks:
        x, y, _ = block["bottomCorner"]
        width, height = block["size"]
        block_patch = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor='gray', alpha=0.5)
        ax.add_patch(block_patch)

    # 绘制每个GU，根据其.connected_nodes的值设置颜色
    for gu in ground_users:
        gu_position = gu.position[:2]  # 只获取 (x, y) 坐标
        connection_type = gu.connected_nodes[0]  # 获取连接类型
        color = connection_types[connection_type]  # 根据连接类型分配颜色
        ax.scatter(*gu_position, color=color, label=f"UAV {connection_type}", s=120, alpha=0.8, edgecolors='w')

    # 设置图形边界和标题
    ax.set_xlim(0, scene["xLength"])
    ax.set_ylim(0, scene["yLength"])
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_title("GU Visualization by Connection Type")
    
    # 仅绘制一次图例条目，且按连接类型从小到大排序
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(sorted(zip(labels, handles), key=lambda x: int(x[0].split()[-1])))
    ax.legend(unique_labels.values(), unique_labels.keys())
    
    plt.grid(True)
    plt.show()

def visualize_capacity_and_load(gu_capacities_records, uav_load_records, normalize=False): 
    # Calculate time points
    time_points = list(range(len(gu_capacities_records)))

    # Normalize GU Capacity if specified
    if normalize:
        all_capacities = [capacity for record in gu_capacities_records for capacity in record.values()]
        min_val, max_val = min(all_capacities), max(all_capacities)
        gu_capacities_records = [
            {gu_index: (capacity - min_val) / (max_val - min_val) if max_val != min_val else 0 
             for gu_index, capacity in record.items()} 
            for record in gu_capacities_records
        ]

    # Calculate min, max, and mean capacities
    min_capacities = [min(capacities.values()) for capacities in gu_capacities_records]
    max_capacities = [max(capacities.values()) for capacities in gu_capacities_records]
    mean_capacities = [np.mean(list(capacities.values())) for capacities in gu_capacities_records]

    # Get UAV IDs and track their load over time
    uav_ids = sorted(set(uav_id for record in uav_load_records for uav_id in record.keys()))
    uav_loads_over_time = {uav_id: [record.get(uav_id, 0) for record in uav_load_records] for uav_id in uav_ids}

    # Plot GU Capacity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(time_points, min_capacities, label='Min Capacity', marker='o', color='blue')
    ax1.plot(time_points, max_capacities, label='Max Capacity', marker='o', color='red')
    ax1.plot(time_points, mean_capacities, label='Mean Capacity', marker='o', color='green')
    ax1.set_xlabel('Time (New UAV Found)')
    ax1.set_ylabel('Capacity' + (' (Normalized)' if normalize else ''))
    ax1.set_title('GU Capacity Over Time')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xticks(time_points)  # Set x-axis to display only integer time points

    # Plot UAV Load Distribution
    bar_width = 0.35
    bottom = np.zeros(len(time_points))
    colors = plt.cm.Paired(np.linspace(0, 1, len(uav_ids))) 
    for i, uav_id in enumerate(uav_ids):
        ax2.bar(time_points, uav_loads_over_time[uav_id], bottom=bottom, label=f'UAV {uav_id}', color=colors[i])
        bottom += np.array(uav_loads_over_time[uav_id]) 

    ax2.set_xlabel('Time (New UAV Found)')
    ax2.set_ylabel('Number of GUs')
    ax2.set_title('UAV Load Distribution Over Time')
    ax2.legend(title="UAV ID")
    ax2.grid(True, axis='y')
    ax2.set_xticks(time_points)  # Set x-axis to display only integer time points

    plt.tight_layout()
    plt.show()



def visualize_scores(reward_track, RS_track, best_reward_track, best_RS_track):
    # 确定episode数
    episodes = np.arange(len(reward_track))
    
    fig, ax = plt.subplots(figsize=(14, 7))

    # 折线图：reward_track 和 RS_track
    ax.plot(episodes, reward_track, label='Reward Track', color='blue', linestyle='-', marker='o')
    ax.plot(episodes, RS_track, label='RS Track', color='green', linestyle='-', marker='o')

    # 柱状图：best_reward_track 和 best_RS_track
    width = 0.4  # 定义柱状图的宽度
    ax.bar(episodes - width/2, best_reward_track, width=width, color='blue', alpha=0.5, label='Best Reward Track')
    ax.bar(episodes + width/2, best_RS_track, width=width, color='green', alpha=0.5, label='Best RS Track')

    # 添加图例和标签
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Reward and RS Tracks with Best Scores')
    ax.legend()

    plt.grid(True)
    plt.show()

def visualize_best_scores(best_reward_track, best_RS_track):
    # 确定 episode 数
    episodes = np.arange(len(best_reward_track))

    fig, ax = plt.subplots(figsize=(14, 7))

    # 使用折线图绘制 best_reward_track 和 best_RS_track
    ax.plot(episodes, best_reward_track, label='Best Reward Track', color='blue', linestyle='-', marker='o')
    ax.plot(episodes, best_RS_track, label='Best RS Track', color='green', linestyle='-', marker='o')

    # 添加图例和标签
    ax.set_xlabel('Episode')
    ax.set_ylabel('Best Score')
    ax.set_title('Best Reward and RS Tracks Over Episodes')
    ax.legend()

    plt.grid(True)
    plt.show()

def visualize_simulation(uav_connections_TD, gu_capacity_TD, num_uavs):
    # 初始化存储每个时间步的最小和平均容量
    min_capacity_over_time = []
    avg_capacity_over_time = []
    uav_connections_over_time = []

    # 遍历每个时间步的数据
    for step, (gu_to_uav_connections, gu_to_bs_capacity) in enumerate(zip(uav_connections_TD, gu_capacity_TD)):
        # 确保gu_to_bs_capacity是数值列表
        if isinstance(gu_to_bs_capacity, dict):
            capacities = list(gu_to_bs_capacity.values())
        else:
            capacities = gu_to_bs_capacity  # 如果已经是列表，则直接使用

        # 计算当前时间步的最小和平均容量
        min_capacity = np.min(capacities)
        avg_capacity = np.mean(capacities)
        min_capacity_over_time.append(min_capacity)
        avg_capacity_over_time.append(avg_capacity)

        # 将gu_to_uav_connections转换为整数值的列表
        gu_to_uav_connections = {k: v[0] if isinstance(v, list) else v for k, v in gu_to_uav_connections.items()}

        # 统计每个UAV的GU连接数量
        uav_connection_counts = [sum(1 for uav in gu_to_uav_connections.values() if uav == i) for i in range(num_uavs)]
        uav_connections_over_time.append(uav_connection_counts)

    # 转置数据以便绘制堆叠柱形图
    uav_connections_over_time = np.array(uav_connections_over_time).T

    # 可视化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 绘制GU到BS的capacity的最小值和平均值折线图
    time_steps = np.arange(len(uav_connections_TD))
    ax1.plot(time_steps, min_capacity_over_time, label="Min Capacity")
    ax1.plot(time_steps, avg_capacity_over_time, label="Avg Capacity")
    ax1.set_title("GU to BS Capacity Over Time")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Capacity")
    ax1.legend()
    ax1.set_xticks(time_steps)  # 设置x轴刻度为整数

    # 绘制堆叠柱形图表示每个时间步的UAV连接数量
    bottom = np.zeros(len(uav_connections_TD))  # 初始化底部位置为0
    for i in range(num_uavs):
        ax2.bar(time_steps, uav_connections_over_time[i], bottom=bottom, label=f"UAV {i}")
        bottom += uav_connections_over_time[i]  # 更新底部位置，堆叠下一个UAV的值

    ax2.set_title("Number of GUs Connected to Each UAV Over Time")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Total Number of GUs Connected")
    ax2.legend(loc="upper left")
    ax2.set_xticks(time_steps)  # 设置x轴刻度为整数

    plt.tight_layout()
    plt.show()

def visualize_simulation_together(uav_connections_TD, gu_capacity_TD, num_uavs):
    # 初始化存储每个时间步的最小和平均容量
    min_capacity_over_time = []
    avg_capacity_over_time = []
    uav_connections_over_time = []

    # 遍历每个时间步的数据
    for step, (gu_to_uav_connections, gu_to_bs_capacity) in enumerate(zip(uav_connections_TD, gu_capacity_TD)):
        # 确保gu_to_bs_capacity是数值列表
        if isinstance(gu_to_bs_capacity, dict):
            capacities = list(gu_to_bs_capacity.values())
        else:
            capacities = gu_to_bs_capacity  # 如果已经是列表，则直接使用

        # 计算当前时间步的最小和平均容量
        min_capacity = np.min(capacities)
        avg_capacity = np.mean(capacities)
        min_capacity_over_time.append(min_capacity)
        avg_capacity_over_time.append(avg_capacity)

        # 将gu_to_uav_connections转换为整数值的列表
        gu_to_uav_connections = {k: v[0] if isinstance(v, list) else v for k, v in gu_to_uav_connections.items()}

        # 统计每个UAV的GU连接数量
        uav_connection_counts = [sum(1 for uav in gu_to_uav_connections.values() if uav == i) for i in range(num_uavs)]
        uav_connections_over_time.append(uav_connection_counts)

    # 转置数据以便绘制堆叠柱形图
    uav_connections_over_time = np.array(uav_connections_over_time).T

    # 可视化
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # 绘制GU到BS的capacity的最小值和平均值折线图
    time_steps = np.arange(len(uav_connections_TD))
    ax1.plot(time_steps, min_capacity_over_time, label="Min Capacity", color="blue", marker="o")
    ax1.plot(time_steps, avg_capacity_over_time, label="Avg Capacity", color="green", marker="x")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Capacity", color="black")
    ax1.legend(loc="upper left")
    ax1.set_xticks(time_steps)  # 设置x轴刻度为整数

    # 使用相同的x轴，增加堆叠柱状图
    ax2 = ax1.twinx()
    ax2.set_ylabel("Total Number of GUs Connected")
    bottom = np.zeros(len(uav_connections_TD))

    for i in range(num_uavs):
        ax2.bar(time_steps, uav_connections_over_time[i], bottom=bottom, label=f"UAV {i}", alpha=0.6)
        bottom += uav_connections_over_time[i]

    ax2.legend(loc="upper right")
    ax2.set_xticks(time_steps)  # 设置x轴刻度为整数

    plt.title("GU to BS Capacity and UAV Connections Over Time")
    plt.tight_layout()
    plt.show()
