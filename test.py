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
# print_node(BS_nodes)

set_connected_edges("101101", UAV_nodes, BS_nodes)
# UAV_nodes[0].add_connection(2)

# print_node(UAV_nodes)
# print_node(BS_nodes)

gu = generate_nodes(2, 0)
# print_node(gu)

blocks = scene_data['blocks']
scene = scene_data['scenario']
UAVInfo = scene_data['UAV']
baseStation = scene_data['baseStation']
nodeNumber = scene_data['nodeNumber']


# print(move_gu_and_update_connections(gu, blocks, scene['xLength'], scene['yLength'], 5, UAV_nodes, UAVInfo))

# print_node(gu)

# move_gu_and_update_connections(gu, blocks, scene['xLength'], scene['yLength'], 5, UAV_nodes, UAVInfo)
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

from visualization_functions import scene_visualization, visualize_simulation_with_baseline
# scene_visualization(gu, UAV_nodes, BS_nodes, scene_data)

# for i in range(0,10,5):
#     print(i)

uav_connections_TD = [{0: [0], 1: [2], 2: [1], 3: [1], 4: [1], 5: [2]}, {0: [0], 1: [1], 2: [2], 3: [1], 4: [1], 5: [0]}, {0: [0], 1: [0], 2: [1], 3: [1], 4: [2], 5: [2]}, {0: [2], 1: [0], 2: [0], 3: [1], 4: [2], 5: [1]}, {0: [2], 1: [2], 2: [1], 3: [1], 4: [0], 5: [1]}, {0: [1], 1: [1], 2: [2], 3: [1], 4: [0], 5: [1]}, {0: [2], 1: [1], 2: [2], 3: [0], 4: [0], 5: [1]}, {0: [2], 1: [0], 2: [2], 3: [2], 4: [1], 5: [0]}, {0: [1], 1: [2], 2: [1], 3: [0], 4: [1], 5: [0]}, {0: [0], 1: [1], 2: [0], 3: [2], 4: [0], 5: [2]}]
gu_capacity_TD = [{0: 9094046669.148972, 1: 10787446568.653963, 2: 8626226531.371231, 3: 9068331138.748259, 4: 9068331138.748259, 5: 10341797459.27825}, {0: 9318226345.385986, 1: 9407560210.08793, 2: 8422425824.900405, 3: 9890298946.68055, 4: 9222988482.580143, 5: 9318226345.385986}, {0: 9681487217.675041, 1: 9627084182.122953, 2: 10029220478.498863, 3: 10029220478.498863, 4: 10029220478.498863, 5: 10029220478.498863}, {0: 9444948129.321573, 1: 9444948129.321573, 2: 9444948129.321573, 3: 10209020147.059689, 4: 9444948129.321573, 5: 10171234989.038868}, {0: 10214609493.875586, 1: 10214609493.875586, 2: 10214609493.875586, 3: 9849978021.163067, 4: 9700567146.696098, 5: 10214609493.875586}, {0: 9767297185.839523, 1: 10367135193.991825, 2: 9071143795.246647, 3: 9784691263.870827, 4: 10214609493.875586, 5: 10485363551.813494}, {0: 9595614339.958605, 1: 8739452507.619184, 2: 10747099571.151974, 3: 9318226345.385986, 4: 9318226345.385986, 5: 8739452507.619184}, {0: 10689610970.66253, 1: 10546117370.66796, 2: 10547647323.38575, 3: 10799167222.696709, 4: 9817334373.909468, 5: 10551427334.272799}, {0: 10211306016.659771, 1: 9140122819.163834, 2: 10229183670.422184, 3: 10132218752.231604, 4: 10399519979.391987, 5: 10132218752.231604}, {0: 10814339392.54353, 1: 10516295589.003399, 2: 10083172166.191685, 3: 9832134478.351082, 4: 10131395102.648333, 5: 8230679381.635712}]
baseline_uav_connections_TD = [{0: [2], 1: [0], 2: [1], 3: [1], 4: [1], 5: [0]}, {0: [2], 1: [0], 2: [1], 3: [0], 4: [1], 5: [0]}, {0: [2], 1: [0], 2: [1], 3: [1], 4: [0], 5: [1]}, {0: [1], 1: [2], 2: [2], 3: [1], 4: [0], 5: [0]}, {0: [2], 1: [1], 2: [1], 3: [0], 4: [0], 5: [1]}, {0: [1], 1: [0], 2: [0], 3: [0], 4: [2], 5: [1]}, {0: [1], 1: [1], 2: [0], 3: [2], 4: [2], 5: [1]}, {0: [0], 1: [1], 2: [2], 3: [0], 4: [2], 5: [1]}, {0: [0], 1: [0], 2: [2], 3: [2], 4: [1], 5: [2]}, {0: [0], 1: [1], 2: [0], 3: [1], 4: [0], 5: [2]}]
baseline_gu_capacity_TD = [{0: 8552788612.873585, 1: 10552778402.113384, 2: 8801594152.365793, 3: 8801594152.365793, 4: 4838.560833189523, 5: 9924952408.586994}, {0: 8552788612.873585, 1: 10569916963.236704, 2: 8801594152.365793, 3: 9743004098.004894, 4: 2667.3516552783913, 5: 9839726263.85189}, {0: 8552788612.873585, 1: 10555435273.166752, 2: 8801594152.365793, 3: 8801594152.365793, 4: 8932937177.418556, 5: 8801594152.365793}, {0: 8500646263.939202, 1: 8552788612.873585, 2: 8552788612.873585, 3: 8801594152.365793, 4: 8882364598.31595, 5: 10468798979.11942}, {0: 1268.4415033818452, 1: 8801594152.365793, 2: 3685.686296015282, 3: 9871763865.212156, 4: 10518332136.943672, 5: 2445.38693751066}, {0: 8801594152.365793, 1: 9029633188.910185, 2: 10692018236.401808, 3: 9562344118.355179, 4: 8552788612.873585, 5: 2340.603686116188}, {0: 8801594152.365793, 1: 8801594152.365793, 2: 10743300190.328861, 3: 8552788612.873585, 4: 8552788612.873585, 5: 5622.721985983665}, {0: 3091.849168298404, 1: 1201.0340659156604, 2: 1866.4547716858413, 3: 2578.641193992535, 4: 8552788612.873585, 5: 6193.7355274492165}, {0: 2768.242032710947, 1: 10128967778.467146, 2: 1874.0625560422698, 3: 8552788612.873585, 4: 3090.8301153428083, 5: 8552788612.873585}, {0: 10258597080.021305, 1: 8801594152.365793, 2: 10642292105.562862, 3: 2088.0483672801597, 4: 10758260513.443367, 5: 8552788612.873585}]
num_UAV = 3

# gu_to_uav_connections = {k: v[0] if isinstance(v, list) else v for k, v in gu_to_uav_connections.items()}
# baseline_gu_to_uav_connections = {k: v[0] if isinstance(v, list) else v for k, v in baseline_gu_to_uav_connections.items()}


# visualize_simulation_with_baseline(uav_connections_TD, gu_capacity_TD, baseline_uav_connections_TD, baseline_gu_capacity_TD, num_UAV)

import json

# Load scene data from JSON file
with open('scene_data_hard.json', 'r') as file:
# with open('scene_data_simple.json', 'r') as file:
# with open('scene_data.json', 'r') as file:
# with open('scene_data_mid.json', 'r') as file:
    scene_data = json.load(file)

blocks = scene_data['blocks']
scene = scene_data['scenario']
UAVInfo = scene_data['UAV']
baseStation = scene_data['baseStation']
nodeNumber = scene_data['nodeNumber']

num_GU = nodeNumber['GU']
num_UAV = nodeNumber['UAV']
num_BS = len(scene_data['baseStation'])

# ---------
from node_functions import generate_nodes, print_node, move_ground_users, get_nodes_position
# 0-GU, 1-UAV, 2-BS
ground_users = generate_nodes(num_GU, 0)

BS_nodes = generate_nodes(num_BS, 2)

for i in range(num_BS):
    BS_nodes[i].set_position((baseStation[i]['bottomCorner'][0], baseStation[i]['bottomCorner'][1], baseStation[i]['height'][0]))

# print_node(ground_users)
from node_functions import add_or_remove_gu
# add_or_remove_gu(ground_users)

GU_position_TD = []
# max_movement_distance = 5
max_movement_distance = 30

# print(scene_data)
# print("--")
# print(scene_data["blocks"])


scene_visualization(ground_users, UAV_nodes=None, air_base_station=BS_nodes, scene_info=scene_data)

for i in range(10):
    move_ground_users(ground_users, blocks, scene['xLength'], scene['yLength'], 2000)

for i in range(50):
    add_or_remove_gu(ground_users)
    move_ground_users(ground_users, blocks, scene['xLength'], scene['yLength'], max_movement_distance)
    # print(get_nodes_position(ground_users))

    GU_position_TD.append(get_nodes_position(ground_users))

    # scene_visualization(ground_users, UAV_nodes=None, air_base_station=None, scene_info=scene_data)

# print(GU_position_TD)
import pandas as pd
df = pd.DataFrame(GU_position_TD)
# # df.to_csv("ground_user_positions_for_simple_scene_50_stable.csv", index=False)
# df.to_csv("ground_user_positions_for_mid_scene_50_stable.csv", index=False)

# df.to_csv("ground_user_positions_for_simple_scene_50_dynamic.csv", index=False)
# df.to_csv("ground_user_positions_for_mid_scene_50_dynamic.csv", index=False)
# df.to_csv("ground_user_positions_for_hard_scene_50_stable.csv", index=False)
# df.to_csv("ground_user_positions_for_hard_scene_50_dynamic.csv", index=False)

# ground_users_positions_simple = pd.read_csv("ground_user_positions_for_simple_scene_50_dynamic.csv")
# ground_users_positions_simple = pd.read_csv("ground_user_positions_for_simple_scene_50_stable.csv")

# print(ground_users_positions_simple.iloc[0])
# print(len(ground_users_positions_simple.iloc[0]))
# print(ground_users_positions_simple.iloc[0,0])

# def get_ground_user_positions_for_a_time(gu_position_information, current_time):
#     ground_users_count = len(gu_position_information.iloc[current_time])
#     ground_users = generate_nodes(ground_users_count, 0)

#     for i in range(ground_users_count):
#         value = gu_position_information.iloc[current_time, i]
#         try:
#             position = tuple(map(float, value.strip("()").split(',')))
#             ground_users[i].set_position(position)
#         except ValueError:
#             print(f"Error: Invalid position string {value}")
    
#     return ground_users

# gu_test = get_ground_user_positions_for_a_time(ground_users_positions_simple, 0)
# print_node(gu_test, -1, True)

import ast
# read_result_test= pd.read_csv("experiment_result_mid.csv")
# read_result_test= pd.read_csv("experiment_result_mid_dynamic.csv")
# read_result_test= pd.read_csv("experiment_result_mid_stable_hyper2.csv")
read_result_test= pd.read_csv("experiment_result_mid_stable_hyper3.csv")

read_uav_connections_TD = read_result_test["UAV Connections"].apply(ast.literal_eval).tolist()
read_uav_gu_capacity_TD = read_result_test["GU Capacity"].apply(ast.literal_eval).tolist()


read_baseline_1_result= pd.read_csv("experiment_result_mid_baseline_1_stable.csv")
# read_baseline_1_result= pd.read_csv("experiment_result_mid_baseline_1_dynamic.csv")

read_baseline_1_uav_connections_TD = read_baseline_1_result["UAV Connections"].apply(ast.literal_eval).tolist()
read_baseline_1_uav_gu_capacity_TD = read_baseline_1_result["GU Capacity"].apply(ast.literal_eval).tolist()

read_baseline_2_result= pd.read_csv("experiment_result_mid_baseline_2_stable.csv")
# read_baseline_2_result= pd.read_csv("experiment_result_mid_baseline_2_dynamic.csv")

read_baseline_2_uav_connections_TD = read_baseline_2_result["UAV Connections"].apply(ast.literal_eval).tolist()
read_baseline_2_uav_gu_capacity_TD = read_baseline_2_result["GU Capacity"].apply(ast.literal_eval).tolist()

# visualize_simulation_with_baseline(read_uav_connections_TD, read_uav_gu_capacity_TD,read_baseline_1_uav_connections_TD, read_baseline_1_uav_gu_capacity_TD, 5)
# visualize_simulation_with_baseline(read_uav_connections_TD, read_uav_gu_capacity_TD,read_baseline_2_uav_connections_TD, read_baseline_2_uav_gu_capacity_TD, 5)
from visualization_functions import visualize_simulation_with_multiple_baselines, visualize_simulation_with_multiple_baselines_styled

visualize_simulation_with_multiple_baselines(read_uav_connections_TD, read_uav_gu_capacity_TD, read_baseline_1_uav_connections_TD, read_baseline_1_uav_gu_capacity_TD, read_baseline_2_uav_connections_TD, read_baseline_2_uav_gu_capacity_TD, 5)
visualize_simulation_with_multiple_baselines_styled(read_uav_connections_TD, read_uav_gu_capacity_TD, read_baseline_1_uav_connections_TD, read_baseline_1_uav_gu_capacity_TD, read_baseline_2_uav_connections_TD, read_baseline_2_uav_gu_capacity_TD, 5, 1)
visualize_simulation_with_multiple_baselines_styled(read_uav_connections_TD, read_uav_gu_capacity_TD, read_baseline_1_uav_connections_TD, read_baseline_1_uav_gu_capacity_TD, read_baseline_2_uav_connections_TD, read_baseline_2_uav_gu_capacity_TD, 5, 3)
visualize_simulation_with_multiple_baselines_styled(read_uav_connections_TD, read_uav_gu_capacity_TD, read_baseline_1_uav_connections_TD, read_baseline_1_uav_gu_capacity_TD, read_baseline_2_uav_connections_TD, read_baseline_2_uav_gu_capacity_TD, 5, 5)
