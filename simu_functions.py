import numpy as np
from functions.calculate_data_rate import calculate_data_rate

# from find_topo import get_RS, measure_overload
from key_functions.quantify_topo import get_RS_with_GU, measure_overload_with_GU

def calculate_current_topology_metrics(ground_users, gu_to_uav_connections, uav_to_bs_connections, uav_info, cur_UAVMap, UAV_nodes, reward_hyper, scene_info, print_metrics=False):
    # 计算地面用户到基站的容量、UAV到基站的容量和UAV的过载
    gu_to_bs_capacity, uav_to_bs_capacity, uav_overload = calculate_capacity_and_overload(
        ground_users, gu_to_uav_connections, uav_to_bs_connections, uav_info, cur_UAVMap, UAV_nodes, scene_info['blocks']
    )

    # 计算 RS（Resilience Score）
    ResilienceScore = get_RS_with_GU(
        ground_users, gu_to_uav_connections, cur_UAVMap, 
        reward_hyper['DRPenalty'], reward_hyper['BPHopConstraint'], reward_hyper['BPDRConstraint'], 
        reward_hyper['droppedRatio'], reward_hyper['ratioDR'], reward_hyper['ratioBP'], 
        reward_hyper['weightDR'], reward_hyper['weightBP'], reward_hyper['weightNP'], 
        scene_info, gu_to_bs_capacity
    )

    # 计算 OL（Overload Score）
    OverloadScore = measure_overload_with_GU(uav_overload)

    # 计算 reward 分数
    rewardScore = ResilienceScore * OverloadScore

    if print_metrics:
        print("Reward Score:", rewardScore)
        print("Resilience Score:", ResilienceScore)
        print("Overload Score:", OverloadScore)
        print("GU to BS Capacity:", gu_to_bs_capacity)
        print("UAV to BS Capacity:", uav_to_bs_capacity)
        print("UAV Overload:", uav_overload)

    return rewardScore, ResilienceScore, OverloadScore, gu_to_bs_capacity, uav_to_bs_capacity, uav_overload


def calculate_capacity_and_overload(ground_users, gu_to_uav_connections, uav_to_bs_connections, uav_info, cur_UAVMap, UAV_nodes, block_info):
    gu_to_bs_capacity = {}
    
    for gu_index, uav_index in gu_to_uav_connections.items():
        cur_gu = ground_users[gu_index]

        # print(cur_gu)
        # print(gu_index)

        # print(UAV_nodes[uav_index[0]].position)
        # print(cur_gu.position)

        # Calculate the data rate from GU to the connected UAV
        is_blocked = path_is_blocked(block_info, UAV_nodes[uav_index[0]], cur_gu)
        gu_to_uav_data_rate = calculate_data_rate(uav_info, UAV_nodes[uav_index[0]].position, cur_gu.position, is_blocked)

        paths = cur_UAVMap.allPaths.get(uav_index[0], [])
        if paths:
            max_dr_path = max(paths, key=lambda x: x['DR'])
            gu_to_bs_capacity[gu_index] = min(gu_to_uav_data_rate, max_dr_path['DR'])
        else:
            gu_to_bs_capacity[gu_index] = 0

    uav_to_bs_capacity = {}
    for uav_index, paths in cur_UAVMap.allPaths.items():
        if paths:
            best_path = max(paths, key=lambda x: x['DR'])
            uav_to_bs_capacity[uav_index] = {
                'path': best_path['path'],
                'DR': best_path['DR']
            }
        else:
            uav_to_bs_capacity[uav_index] = {
                'path': [],
                'DR': 0
            }

    # uav_overload = {}

    uav_overload = {uav_index: 0 for uav_index in uav_to_bs_capacity.keys()}

    for gu_index, uav_index in gu_to_uav_connections.items():
        gu_capacity = gu_to_bs_capacity[gu_index]
        uav_path = uav_to_bs_capacity[uav_index[0]]['path']
        uav_path_capacity = uav_to_bs_capacity[uav_index[0]]['DR']

        for uav_idx in uav_path:
            if uav_idx < len(uav_overload):
                uav_overload[uav_idx] += min(gu_capacity, uav_path_capacity)

    return gu_to_bs_capacity, uav_to_bs_capacity, uav_overload       

from functions.path_is_blocked import path_is_blocked

def get_gu_to_uav_connections(ground_users, UAV_nodes, UAVInfo, blocks):
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
        gu_to_uav[gu_index] = [best_uav]  # 使用列表格式表示最佳UAV

    return gu_to_uav

import matplotlib.pyplot as plt
def visualize_all_gu_capacity(all_gu_capacity):
    """
    输入是一个包含每个时间点 GU_capacity 的列表，生成容量变化的可视化结果
    :param all_gu_capacity: List of dicts, each dict contains GU_capacity at a given time point
    """
    # 准备数据进行可视化
    time_points = list(range(len(all_gu_capacity)))
    
    # 获取所有时间点出现过的所有GU
    all_keys = set()
    for gu_capacity in all_gu_capacity:
        all_keys.update(gu_capacity.keys())

    # 可视化
    plt.figure(figsize=(10, 6))
    for key in sorted(all_keys):
        values = []
        for gu_capacity in all_gu_capacity:
            if key in gu_capacity:
                values.append(gu_capacity[key])
            else:
                values.append(None)  # 如果某时间点GU不存在，则设为None

        plt.plot(time_points, values, label=f'GU {key}', marker='o')  # 使用marker区分点

    plt.xlabel('Time Points')
    plt.ylabel('GU Capacity')
    plt.title('GU Capacity Over Time')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 将图例放在右上角外
    plt.grid(True)
    plt.show()

def visualize_all_UAV_overload(all_UAV_overload):
    time_points = list(range(len(all_UAV_overload)))
    keys = all_UAV_overload[0].keys()

    # 可视化
    plt.figure(figsize=(10, 6))
    for key in keys:
        values = [gu_capacity[key] for gu_capacity in all_UAV_overload]
        plt.plot(time_points, values, label=f'GU {key}')

    plt.xlabel('Time Points')
    plt.ylabel('UAV Overload')
    plt.title('UAV Overload Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_uav_capacity(all_uav_capacity):
    """
    可视化 UAV_capacity 中的 DR 数据
    :param all_uav_capacity: List of dicts, each dict contains UAV_capacity at a given time point
    """
    # 准备数据进行可视化
    time_points = list(range(len(all_uav_capacity)))
    keys = all_uav_capacity[0].keys()

    # 可视化
    plt.figure(figsize=(10, 6))
    for key in keys:
        values = [uav_capacity[key]['DR'] for uav_capacity in all_uav_capacity]
        plt.plot(time_points, values, label=f'UAV {key}')

    plt.xlabel('Time Points')
    plt.ylabel('DR (Data Rate)')
    plt.title('UAV Data Rate Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_metrics(max_reward_TD, max_RS_TD, max_OL_TD):

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
    plt.plot(time_points, max_OL_TD, label='Max OL TD', marker='^')

    # 添加标签和标题
    plt.xlabel('Time Points')
    plt.ylabel('Scores')
    plt.title('Max Reward TD, Max RS TD, and Max OL TD Over Time')
    plt.legend()
    plt.grid(True)

    # 显示图表
    plt.show()

from gu_movement import move_ground_users
def move_gu_and_update_connections(ground_users, blocks, x_length, y_length, max_movement_distance, UAV_nodes, UAVInfo):
    """
    移动地面用户并重新计算GU到UAV的连接。

    参数:
    ground_users: 地面用户列表。
    blocks: 障碍物列表。
    x_length: 场景的x方向长度。
    y_length: 场景的y方向长度。
    max_movement_distance: 地面用户的最大移动距离。
    UAV_nodes: UAV节点列表。
    UAVInfo: UAV信息字典。

    返回:
    更新后的GU到UAV连接字典。
    """
    
    # 移动地面用户
    move_ground_users(ground_users, blocks, x_length, y_length, max_movement_distance)

    # 重新计算GU到UAV的连接
    gu_to_uav_connections = get_gu_to_uav_connections(ground_users, UAV_nodes, UAVInfo, blocks)
    
    return gu_to_uav_connections
