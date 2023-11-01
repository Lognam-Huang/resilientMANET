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


import gym
from gym import spaces
import numpy as np
import random

class UAVEnvironment(gym.Env):
    def __init__(self):
        super(UAVEnvironment, self).__init__()
        self.action_space = spaces.Discrete(4)  # 上，下，左，右
        self.observation_space = spaces.Box(low=0, high=10, shape=(7,), dtype=np.float32)
        self.reset()

    def reset(self):
        # 初始化GU和UAV的位置
        self.GU_positions = np.random.uniform(0, 10, (5, 2))
        self.UAV_positions = np.random.uniform(0, 10, (2, 2))
        self.ABS_position = np.array([5, 5, 10])  # 假设ABS位于空间中心
        self.state = np.concatenate((self.GU_positions.flatten(), self.UAV_positions.flatten()))
        return self.state

    def step(self, action):
        # 更新UAV的位置
        if action == 0:  # 上
            self.UAV_positions[:, 1] += 0.1
        elif action == 1:  # 下
            self.UAV_positions[:, 1] -= 0.1
        elif action == 2:  # 左
            self.UAV_positions[:, 0] -= 0.1
        elif action == 3:  # 右
            self.UAV_positions[:, 0] += 0.1

        # 计算奖励
        reward = self.calculate_reward()

        # 更新状态
        self.state = np.concatenate((self.GU_positions.flatten(), self.UAV_positions.flatten()))

        # 检查是否结束
        done = False
        if reward > 0.9:
            done = True

        return self.state, reward, done, {}

    def calculate_reward(self):
        # 计算所有GU之间的最低连接质量
        min_quality = float('inf')
        for i in range(5):
            for j in range(i+1, 5):
                quality = self.calculate_connection_quality(self.GU_positions[i], self.GU_positions[j])
                min_quality = min(min_quality, quality)
        return min_quality

    def calculate_connection_quality(self, pos1, pos2):
        # 假设连接质量与距离成反比
        distance = np.linalg.norm(pos1 - pos2)
        quality = 1 / (1 + distance)
        return quality

env = UAVEnvironment()
state = env.reset()

# 随机策略示例
for _ in range(1000):
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    if done:
        break

print("Final Reward:", reward)
