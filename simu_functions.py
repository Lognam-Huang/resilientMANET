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