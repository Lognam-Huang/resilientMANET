# import ipyvolume as ipv

# # 你的数据
# blocks = [
#     {'bottomCorner': [350, 380], 'size': [80, 80], 'height': 100},
#     {'bottomCorner': [350, 10], 'size': [60, 80], 'height': 200},
#     {'bottomCorner': [20, 570], 'size': [100, 80], 'height': 400}
# ]
# scene = {'xLength': 500, 'yLength': 700}
# groundUserPosition = (136.0598323772384, 398.166713839298, 0)

# # 绘制blocks
# for block in blocks:
#     x, y = block['bottomCorner']
#     dx, dy = block['size']
#     dz = block['height']
#     ipv.plot_mesh([x, x+dx, x+dx, x, x, x+dx, x+dx, x],
#                 [y, y, y+dy, y+dy, y, y, y+dy, y+dy], 
#                 [0, 0, 0, 0, dz, dz, dz, dz], 
#                 triangles=[[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 3, 7], [0, 4, 7], [1, 2, 6], [1, 5, 6], [0, 1, 5], [0, 4, 5], [2, 3, 7], [2, 6, 7]], 
#                 color="blue")

# # 绘制groundUser
# ipv.scatter(*groundUserPosition, color="red", marker="sphere", size=5)

# # 显示
# ipv.show()

# import ipyvolume as ipv

# # 你的数据
# blocks = [
#     {'bottomCorner': [350, 380], 'size': [80, 80], 'height': 100},
#     {'bottomCorner': [350, 10], 'size': [60, 80], 'height': 200},
#     {'bottomCorner': [20, 570], 'size': [100, 80], 'height': 400}
# ]
# scene = {'xLength': 500, 'yLength': 700}
# groundUserPosition = (136.0598323772384, 398.166713839298, 0)

# # 绘制blocks
# for block in blocks:
#     x, y = block['bottomCorner']
#     dx, dy = block['size']
#     dz = block['height']
#     ipv.plot_trisurf([x, x+dx, x+dx, x, x, x+dx, x+dx, x], 
#                     [y, y, y+dy, y+dy, y, y, y+dy, y+dy], 
#                     [0, 0, 0, 0, dz, dz, dz, dz], 
#                     triangles=[[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 3, 7], [0, 4, 7], [1, 2, 6], [1, 5, 6], [0, 1, 5], [0, 4, 5], [2, 3, 7], [2, 6, 7]], 
#                     color="blue")

# # 绘制groundUser
# ipv.scatter(*groundUserPosition, color="red", marker="sphere", size=5)

# # 显示
# ipv.show()

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 你的数据，增加了颜色、透明度和标记
# blocks = [
#     {'bottomCorner': [350, 380], 'size': [80, 80], 'height': 100, 'color': (1, 0, 0, 0.5), 'label': 'Block 1'},
#     {'bottomCorner': [350, 10], 'size': [60, 80], 'height': 200, 'color': (0, 1, 0, 0.7), 'label': 'Block 2'},
#     {'bottomCorner': [20, 570], 'size': [100, 80], 'height': 400, 'color': (0, 0, 1, 0.9), 'label': 'Block 3'}
# ]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim([0, 600])
# ax.set_ylim([0, 600])
# ax.set_zlim([0, 600])

# for block in blocks:
#     x, y = block['bottomCorner']
#     dx, dy = block['size']
#     dz = block['height']
#     color = block['color']
#     label = block['label']
    
#     ax.bar3d(x, y, 0, dx, dy, dz, shade=True, color=color)
    
#     # 添加标记文字在块的中心
#     ax.text(x + dx/2, y + dy/2, dz/2, label, color='black', ha='center', va='center')

# plt.show()


# import gym
# from gym import spaces
# import numpy as np
# import random

# class UAVEnvironment(gym.Env):
#     def __init__(self):
#         super(UAVEnvironment, self).__init__()
#         self.action_space = spaces.Discrete(4)  # 上，下，左，右
#         self.observation_space = spaces.Box(low=0, high=10, shape=(7,), dtype=np.float32)
#         self.reset()

#     def reset(self):
#         # 初始化GU和UAV的位置
#         self.GU_positions = np.random.uniform(0, 10, (5, 2))
#         self.UAV_positions = np.random.uniform(0, 10, (2, 2))
#         self.ABS_position = np.array([5, 5, 10])  # 假设ABS位于空间中心
#         self.state = np.concatenate((self.GU_positions.flatten(), self.UAV_positions.flatten()))
#         return self.state

#     def step(self, action):
#         # 更新UAV的位置
#         if action == 0:  # 上
#             self.UAV_positions[:, 1] += 0.1
#         elif action == 1:  # 下
#             self.UAV_positions[:, 1] -= 0.1
#         elif action == 2:  # 左
#             self.UAV_positions[:, 0] -= 0.1
#         elif action == 3:  # 右
#             self.UAV_positions[:, 0] += 0.1

#         # 计算奖励
#         reward = self.calculate_reward()

#         # 更新状态
#         self.state = np.concatenate((self.GU_positions.flatten(), self.UAV_positions.flatten()))

#         # 检查是否结束
#         done = False
#         if reward > 0.9:
#             done = True

#         return self.state, reward, done, {}

#     def calculate_reward(self):
#         # 计算所有GU之间的最低连接质量
#         min_quality = float('inf')
#         for i in range(5):
#             for j in range(i+1, 5):
#                 quality = self.calculate_connection_quality(self.GU_positions[i], self.GU_positions[j])
#                 min_quality = min(min_quality, quality)
#         return min_quality

#     def calculate_connection_quality(self, pos1, pos2):
#         # 假设连接质量与距离成反比
#         distance = np.linalg.norm(pos1 - pos2)
#         quality = 1 / (1 + distance)
#         return quality

# env = UAVEnvironment()
# state = env.reset()

# # 随机策略示例
# for _ in range(1000):
#     action = env.action_space.sample()
#     state, reward, done, _ = env.step(action)
#     if done:
#         break

# print("Final Reward:", reward)

# import pandas as pd
# from itertools import combinations

# def create_graph_table_enhanced(n):
#     # Generate all possible edges for a complete graph of n nodes
#     edges = list(combinations(range(n), 2))

#     # Generate row names: all possible states for edges (0 or 1 for each edge)
#     num_edges = len(edges)
#     row_names = [format(i, '0' + str(num_edges) + 'b') for i in range(2 ** num_edges)]

#     # Generate column names: + and - for each edge
#     col_names = ['+(' + str(i) + ',' + str(j) + ')' for i, j in edges] + \
#                 ['-(' + str(i) + ',' + str(j) + ')' for i, j in edges]

#     # Create a DataFrame
#     table = pd.DataFrame(0, index=row_names, columns=col_names)

#     # Update the table to reflect the feasibility of each action in each state
#     for row_name in row_names:
#         # For each edge state in the row
#         for i, edge_state in enumerate(row_name):
#             # If the edge is 0 (not connected), '-' action is not feasible
#             if edge_state == '0':
#                 table.at[row_name, '-(' + str(edges[i][0]) + ',' + str(edges[i][1]) + ')'] = -1
#             # If the edge is 1 (connected), '+' action is not feasible
#             else:
#                 table.at[row_name, '+(' + str(edges[i][0]) + ',' + str(edges[i][1]) + ')'] = -1
#     return table

# # Example usage:
# graph_table = create_graph_table_enhanced(3)
# print(graph_table)

# from itertools import combinations

# def get_connected_edges(edge_state):
#     # Calculate the number of nodes from the length of the edge state
#     # This is a reverse calculation from the number of edges to the number of nodes
#     num_edges = len(edge_state)
#     n = int((1 + (1 + 8 * num_edges)**0.5) / 2)  # Solve quadratic equation for number of nodes

#     # Generate all possible edges for a complete graph of n nodes
#     edges = list(combinations(range(n), 2))

#     # Determine which edges are connected based on the edge_state string
#     connected_edges = [edges[i] for i, state in enumerate(edge_state) if state == '1']

#     return connected_edges

# # Example usage for edge state "111011"
# connected_edges = get_connected_edges("111011")
# print(connected_edges)

# from itertools import combinations

# def get_connected_edges(edge_state, m, n):
#     total_nodes = m + n
#     edges = list(combinations(range(total_nodes), 2))
#     connected_edges = [edges[i] for i, state in enumerate(edge_state) if state == '1']
    
#     # Initialize connection lists for each node
#     UAVConnections = {i: [] for i in range(m)}  # for m UAV nodes
#     ABSConnections = {i: [] for i in range(n)}  # for n ABS nodes
    
#     # Fill in the connections based on the connected edges
#     for edge in connected_edges:
#         a, b = edge
        
#         # Both nodes are UAVs
#         if a < m and b < m:
#             UAVConnections[a].append(b)
#             UAVConnections[b].append(a)
#         # One node is UAV (a) and other is ABS (b)
#         elif a < m and b >= m:
#             ABSConnections[b - m].append(a)  # Keep a as is for UAV
#         # One node is UAV (b) and other is ABS (a)
#         elif b < m and a >= m:
#             ABSConnections[a - m].append(b)  # Keep b as is for UAV
    
#     # Now call the set_connection method for each UAV and ABS node
#     for idx, connections in UAVConnections.items():
#         UAVNodes[idx].set_connection(sorted(list(set(connections))))  # Remove duplicates and sort
    
#     for idx, connections in ABSConnections.items():
#         ABSNodes[idx].set_connection(sorted(list(set(connections))))  # Remove duplicates and sort

# # Define example UAV and ABS node classes with a set_connection method
# class Node:
#     def __init__(self, id):
#         self.id = id
    
#     def set_connection(self, connections):
#         print(f"Node {self.id} set connections with {connections}")

# # Suppose you have 3 UAV nodes and 1 ABS node
# UAVNodes = [Node(i) for i in range(3)]
# ABSNodes = [Node(0)]  # Only one ABS node, but its id within ABSNodes is 0

# # Use the function with example state "111011", m=3 UAV nodes, n=1 ABS node
# get_connected_edges("111011", 3, 1)


# from itertools import combinations

# def get_new_state(state, action):
#     # Convert the state from string to list for easy manipulation
#     state_list = list(state)
    
#     # Parse the action to determine the type and edge
#     action_type, edge_info = action[0], action[1:]  # '+' or '-', and the edge tuple '(i,j)'
#     i, j = map(int, edge_info.strip('()').split(','))  # Extract edge indices from the action
    
#     # Find the index of the edge in the state string
#     n = int((1 + (1 + 8 * len(state))**0.5) / 2)  # Calculate n from the length of state string
#     edges = list(combinations(range(n), 2))  # Generate the list of edges
#     edge_index = edges.index((i, j))  # Get the index of the edge in the list
    
#     # Update the state based on the action type
#     if action_type == '+':  # Add edge
#         state_list[edge_index] = '1' if state_list[edge_index] == '0' else state_list[edge_index]
#     elif action_type == '-':  # Remove edge
#         state_list[edge_index] = '0' if state_list[edge_index] == '1' else state_list[edge_index]
    
#     # Convert the state list back to string and return
#     return ''.join(state_list)

# # Example usage
# print(get_new_state("000000", "+(1,2)"))  # Should set the edge (0,1) to '1' if it's '0'
# print(get_new_state("000001", "-(0,1)"))  # Should set the edge (0,1) to '0' if it's '1'

# for i in range(5): 
#   a = 5
#   print(i)
#   while True:
#     a += 1
#     if i == 2: 
#       break
#     if a == 7:
#       break
#     print("a="+str(a))

zero_str = '0' * 6
print(zero_str)