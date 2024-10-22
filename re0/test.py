from itertools import combinations

# print(list(combinations(range(5), 2)))

# for a in range(3):
#     print(a)
#     for b in range (5):
#         print(b)
#         if b == 2:
#             break

from connectivity_finding import generate_adjacent_states, generate_random_binary_string, set_connected_edges, get_backhaul_connection, get_RS, select_all_drops
# a = generate_adjacent_states("100")
# print(a)
# print(len(a))

# b = generate_random_binary_string("00000000")
# print(b)

from node_functions import generate_nodes, print_node, move_gu_and_update_connections
UAV_nodes = generate_nodes(3, 1, default_height=100)
BS_nodes = generate_nodes(1, 2, default_height=200)

UAV_nodes[0].set_position((10,10,15))
UAV_nodes[1].set_position((20,5,15))
UAV_nodes[2].set_position((15,20,15))

BS_nodes[0].set_position((20,20,0))

# set_connected_edges("000000", UAV_nodes, BS_nodes)
# print_node(UAV_nodes, onlyPosition=True)
# print_node(UAV_nodes)
# print("ss")
# print_node(BS_nodes, onlyPosition=True)

import json
with open('scene_data_simple.json', 'r') as file:
    scene_data = json.load(file)

# from BackhaulPaths import 
# print(get_backhaul_connection("111111", UAV_nodes, BS_nodes, scene_data))
# bc = get_backhaul_connection("000000", UAV_nodes, BS_nodes, scene_data)
# bc = get_backhaul_connection("000101", UAV_nodes, BS_nodes, scene_data)
# bc = get_backhaul_connection("000010", UAV_nodes, BS_nodes, scene_data)
import math
def get_edges_from_state(state):
    """
    检查state的长度是否符合无向图的边数公式，并返回对应的边列表
    
    :param state: 表示边的01字符串
    :return: 如果state长度合法，返回对应的边列表；否则返回错误提示
    """
    L = len(state)  # 获取state的长度
    
    # 使用公式 n = (1 + sqrt(1 + 8 * L)) / 2
    n = (1 + math.sqrt(1 + 8 * L)) / 2
    
    # 检查n是否是整数
    if not n.is_integer():
        return "State 长度不合法，不能表示一个完整的无向图"
    
    n = int(n)  # 将n转换为整数节点数
    edges = []  # 存储边的列表
    node_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]  # 生成所有节点对

    for index, pair in enumerate(node_pairs):
        if state[index] == '1':  # 如果对应位置为1，表示有边
            edges.append(pair)
    
    return edges

gs = get_edges_from_state("000101")
# for start_node, end_node in gs:
    # print(start_node)
    # print(end_node)

from connectivity_finding import set_connected_edges
# print_node(UAV_nodes)
print_node(BS_nodes)

set_connected_edges("101101", UAV_nodes, BS_nodes)
# UAV_nodes[0].add_connection(2)

# print_node(UAV_nodes)
print_node(BS_nodes)

gu = generate_nodes(2, 0)
# print_node(gu)

blocks = scene_data['blocks']
scene = scene_data['scenario']
UAVInfo = scene_data['UAV']
baseStation = scene_data['baseStation']
nodeNumber = scene_data['nodeNumber']


# print(move_gu_and_update_connections(gu, blocks, scene['xLength'], scene['yLength'], 5, UAV_nodes, UAVInfo))

# print_node(gu)

move_gu_and_update_connections(gu, blocks, scene['xLength'], scene['yLength'], 5, UAV_nodes, UAVInfo)
# print(gu[0].data_rate)
# print(gu[0].data_rate[0])
# print(gu[0].connected_nodes[0])
# print(gu[1].data_rate)
# print_node(gu)

reward_hyper = {
    'DRPenalty': 0.5,
    'BPHopConstraint': 6,
    'BPDRConstraint': 1000000,
    'droppedRatio': 0.5,
    'ratioDR': 0.6,
    'ratioBP': 0.4,
    'weightDR': 0.3,
    'weightBP': 0.4,
    'weightNP': 0.3,
    'overloadConstraint': 10000
}
# print(bc)
# for start_uav_node, paths in bc.allPaths.items():
#     # print(start_uav_node)
#     # print(paths)
#     for path in paths:
#         print(path['path'])
#         print(path['DR'])
#         print(len(path['path']))
# print(max(path['DR'] for path in bc.allPaths[0]))

# print(bc.allPaths[0])
# print(bc.allPaths[0][0])
# print(bc.allPaths[0][0]['path'])
# print(len(bc.allPaths[0][0]['path']))

# print(select_all_drops(bc, 0.9))
# print(get_RS(gu, bc, reward_hyper, scene_data))

from connectivity_finding import disable_bs_edges_in_state, edge_index_in_state, reward
# print(disable_bs_edges_in_state("111111", 1, 2))
# print(edge_index_in_state(2,3,4))

import math
# n = (1 + math.sqrt(1 + 8 * len("00"))) / 2
# print(n)

# print(reward("111111", scene_data, gu, UAV_nodes, BS_nodes, reward_hyper))

# print_node(gu)
# print_node(UAV_nodes)
# print_node(BS_nodes)

from visualization_functions import scene_visualization
# scene_visualization(gu, UAV_nodes, BS_nodes, scene_data)