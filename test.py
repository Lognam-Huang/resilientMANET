# q = {}
# # print(type(q))

# def generate_adjacent_states(state):
#     adjacent_states = []

#     for i in range(len(state)):
#         state_list = list(state)
#         state_list[i] = '1' if state[i] == '0' else '0'
#         adjacent_states.append(''.join(state_list))

#     return adjacent_states


# def process_states(adjacent_states, q_table):

#     print(len(adjacent_states))

#     for state in adjacent_states:
#         if state in q_table:
#             print(f"State: {state}, Score: {q_table[state]}")
#         else:
#             q_table[state] = 1
#             print(f"Added new state {state} with score 1")

# # 示例
# q_table = {'01101': 0.745, '10011': 0.658}
# input_state = '01101'

# adjacent_states = generate_adjacent_states(input_state)
# process_states(adjacent_states, q_table)

# # 查看更新后的 q_table
# print("\nUpdated q_table:")
# print(q_table)

# import random

# def take_action(a, b):
#     # 检查 a 是否为空
#     if not a:
#         return None

#     # 获取所有值非零的键值对
#     non_zero_items = {k: v for k, v in a.items() if v != 0}

#     # 如果所有项的值都为零，返回 None
#     if not non_zero_items:
#         return None

#     if random.random() < b:
#         # 在 b 的概率下，随机选择一个值非零的键值对
#         return random.choice(list(non_zero_items.items()))
#     else:
#         # 在 1-b 的概率下，选择具有最大值的键值对
#         max_key = max(non_zero_items, key=non_zero_items.get)
#         return max_key, non_zero_items[max_key]

# # 示例
# a = {'state1': 0.5, 'state2': 0, 'state3': 0.7, 'state4': 0.3}
# b = 0

# result = take_action(a, b)
# print(result[0])

# import random

# def generate_random_binary_string(input_string):
#     length = len(input_string)
#     random_string = ''.join(random.choice(['0', '1']) for _ in range(length))
#     return random_string

# # 示例
# input_string = '0110101'
# output_string = generate_random_binary_string(input_string)

# print("Input String: ", input_string)
# print("Random Binary String: ", output_string)

# alist = {}
# alist["11"] = 11,32,44
# alist["12"] = 77,22,44
# print(alist["11"][2])

# print(alist.get)
# # max_key = max(alist, key=alist.get)
# max_key = max(alist, key=lambda k: alist[k][1])

# print(max_key)

def ss():
    return 1, 3, 5

s = ss()
print(s)

import json
from functions.get_3D_heatmap import get_3D_heatmap
from functions.generate_users import generate_users
from functions.print_nodes import print_nodes
from functions.scene_visualization import *
import numpy as np



with open('scene_data_simple.json', 'r') as f:
    ini = json.load(f)

groundBaseStation = ini['baseStation']
blocks = ini['blocks']
UAVInfo = ini['UAV']
scene = ini['scenario']

ground_users = generate_users(10, blocks, scene['xLength'], scene['yLength'])
print_nodes(ground_users, True)


min_height = 10  # 最小高度
max_height = 15  # 最大高度

heatmap = get_3D_heatmap(ground_users, blocks, scene, min_height, max_height)

# visualize_2D_heatmap_per_layer(heatmap=heatmap, min_height=min_height, max_height= max_height)

visualize_2D_heatmap_combined(heatmap=heatmap, min_height=min_height, max_height= max_height)
# scene_visualization(ground_users=ground_users, blocks=blocks, scene_info=scene, heatmap=heatmap, min_height=min_height)

# max_users = np.max(heatmap)  # 获取最大的ground_user数
# print("number of users"+str(max_users))
# for x in range(heatmap.shape[0]):
#     for y in range(heatmap.shape[1]):
#         for z in range(heatmap.shape[2]):
#             value = heatmap[x, y, z]
#             if value > 0:  # 如果该点的值大于0
#                 # alpha = (value / max_users) * 0.7  # 根据ground_user的数量调整透明度
#                 print(str([x,y,z]))

# print(heatmap)