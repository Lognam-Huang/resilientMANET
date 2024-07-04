import numpy as np
from functions.calculate_data_rate import calculate_data_rate

def calculate_capacity_and_overload(ground_users, gu_to_uav_connections, uav_to_bs_connections, uav_info, cur_UAVMap, UAV_nodes):
    gu_to_bs_capacity = {}
    for gu_index, uav_index in gu_to_uav_connections.items():
        cur_gu = ground_users[gu_index]

        # print(cur_gu)
        # print(gu_index)

        # print(UAV_nodes[uav_index[0]].position)
        # print(cur_gu.position)

        # Calculate the data rate from GU to the connected UAV
        gu_to_uav_data_rate = calculate_data_rate(uav_info, UAV_nodes[uav_index[0]].position, cur_gu.position, False)

        paths = cur_UAVMap.allPaths.get(uav_index[0], [])
        if paths:
            max_dr_path = max(paths, key=lambda x: x['DR'])

        gu_to_bs_capacity[gu_index] = min(gu_to_uav_data_rate, max_dr_path['DR'])

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

    uav_overload = {}

    uav_overload = {uav_index: 0 for uav_index in uav_to_bs_capacity.keys()}

    for gu_index, uav_index in gu_to_uav_connections.items():
        gu_capacity = gu_to_bs_capacity[gu_index]
        uav_path = uav_to_bs_capacity[uav_index[0]]['path']
        uav_path_capacity = uav_to_bs_capacity[uav_index[0]]['DR']

        for uav_idx in uav_path:
            if uav_idx < len(uav_overload):
                uav_overload[uav_idx] += min(gu_capacity, uav_path_capacity)

    return gu_to_bs_capacity, uav_to_bs_capacity, uav_overload       

import matplotlib.pyplot as plt
def visualize_all_gu_capacity(all_gu_capacity):
    """
    输入是一个包含每个时间点 GU_capacity 的列表，生成容量变化的可视化结果
    :param all_gu_capacity: List of dicts, each dict contains GU_capacity at a given time point
    """
    # 准备数据进行可视化
    time_points = list(range(len(all_gu_capacity)))
    keys = all_gu_capacity[0].keys()

    # 可视化
    plt.figure(figsize=(10, 6))
    for key in keys:
        values = [gu_capacity[key] for gu_capacity in all_gu_capacity]
        plt.plot(time_points, values, label=f'GU {key}')

    plt.xlabel('Time Points')
    plt.ylabel('GU Capacity')
    plt.title('GU Capacity Over Time')
    plt.legend()
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