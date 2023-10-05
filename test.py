# def is_blocked_by_box(A, B, box):
#     # Step 1: 判断两点是否在长方体的同一侧
#     for dim in range(3):
#         if (A[dim] < box['bottomCorner'][dim] and B[dim] < box['bottomCorner'][dim]) or \
#             (A[dim] > box['bottomCorner'][dim] + (box['size'][dim] if dim < 2 else box['height']) and
#             B[dim] > box['bottomCorner'][dim] + (box['size'][dim] if dim < 2 else box['height'])):
#             return False

#     # Step 2: 判断两点之间的线段是否与长方体的任意面相交
#     # 这里只是一个简化的判断，为了真正的判断需要更复杂的计算
#     for dim in range(2):
#         if (A[dim] <= box['bottomCorner'][dim] <= B[dim] or B[dim] <= box['bottomCorner'][dim] <= A[dim]) and \
#             (A[2] <= box['bottomCorner'][2] <= B[2] or B[2] <= box['bottomCorner'][2] <= A[2]):
#             return True
#         if (A[dim] <= box['bottomCorner'][dim] + box['size'][dim] <= B[dim] or
#             B[dim] <= box['bottomCorner'][dim] + box['size'][dim] <= A[dim]) and \
#             (A[2] <= box['bottomCorner'][2] + box['height'] <= B[2] or
#             B[2] <= box['bottomCorner'][2] + box['height'] <= A[2]):
#             return True
#     return False

# def is_blocked(A, B, blocks):
#     for box in blocks:
#         if is_blocked_by_box(A, B, box):
#             return True
#     return False

# # 测试数据
# blocks = [
#     {'bottomCorner': [350, 380, 0], 'size': [80, 80], 'height': 100},
#     {'bottomCorner': [350, 10, 0], 'size': [60, 80], 'height': 200},
#     {'bottomCorner': [20, 570, 0], 'size': [100, 80], 'height': 400}
# ]

# A = [0, 0, 0]
# B = [200,200,0]

# print(is_blocked(A, B, blocks))  # 返回值表示A和B之间是否被长方体阻挡


# class UAVMap:
#     def __init__(self, AllPaths):
#         self.AllPaths = AllPaths
    
#     def quantify_data_rate(self, r):
#         print(self.AllPaths.values())
#         # 1. 获取每个元素的最大DR
#         max_data_rates = [max(paths, key=lambda x: x['DR'])['DR'] if paths else 0 for paths in self.AllPaths.values()]
        
#         print(max_data_rates)
        
#         # 2. 计算所有元素的最小DR和平均DR
#         min_DR = min(max_data_rates)
#         avg_DR = sum(max_data_rates) / len(max_data_rates)
        
#         # 3. 使用公式计算score
#         score = r * min_DR + (1 - r) * avg_DR
#         return score

# # 示例
# uav_map = UAVMap(AllPaths={0: [{'path': [0, 2, 6], 'DR': 1394949843.0551932}, {'path': [0, 6], 'DR': 1902723297.561852}], 1: [{'path': [1, 0, 2, 6], 'DR': 312893737.12986887}, {'path': [1, 0, 6], 'DR': 312893737.12986887}], 2: [{'path': [2, 0, 6], 'DR': 1488013526.2304676}, {'path': [2, 6], 'DR': 1394949843.0551932}], 3: [{'path': [3, 5], 'DR': 2025826703.0208344}], 4: [{'path': [4, 6], 'DR': 2089479536.0437372}]
#                         # 5: []
#                         })

# r = 0.5  # 可以根据需要改变
# score = uav_map.quantify_data_rate(r)
# print(score)


# def quantify_backup_path(AllPaths, hop_constraint, DR_constraint):
#     # 函数内部用于计算hop
#     def hop_count(path):
#         return len(path)

#     # 计算每个起点的最佳DR
#     best_DRs = {}
#     for start, paths in AllPaths.items():
#         filtered_paths = [p for p in paths if hop_count(p['path']) <= hop_constraint and p['DR'] >= DR_constraint]
#         if filtered_paths:
#             best_DRs[start] = max(p['DR'] for p in filtered_paths)
#         else:
#             best_DRs[start] = None
#         # print(filtered_paths)
#         # print(best_DRs)

#     # 计算每条路径的得分
#     total_score = 0
#     max_path_count = max(len(paths) for paths in AllPaths.values())
#     for start, paths in AllPaths.items():
#         for p in paths:
#             if hop_count(p['path']) <= hop_constraint and p['DR'] >= DR_constraint:
#                 if p['DR'] == best_DRs[start]:  # 最佳路径得分为1
#                     total_score += 1
#                 else:
#                     total_score += p['DR'] / best_DRs[start]

#     # 得分总和除以路径的最大值
#     result = total_score / max_path_count
#     return result

# # 示例
# AllPaths = {
#     0: [{'path': [0, 2, 6], 'DR': 1394949843.0551932}, {'path': [0, 6], 'DR': 1902723297.561852}],
#     1: [{'path': [1, 0, 2, 6], 'DR': 312893737.12986887}, {'path': [1, 0, 6], 'DR': 312893737.12986887}],
#     2: [{'path': [2, 0, 6], 'DR': 1488013526.2304676}, {'path': [2, 6], 'DR': 1394949843.0551932}],
#     3: [{'path': [3, 5], 'DR': 2025826703.0208344}],
#     4: [{'path': [4, 6], 'DR': 2089479536.0437372}]
# }

# result = quantify_backup_path(AllPaths, 4, 1000000000)
# print(result)


# import random

# def select_ten_percent(N):
#     num_samples = int(N * 0.1)  # 计算10%的大小
#     return random.sample(range(N+1), num_samples)

# N = 9
# selected_numbers = select_ten_percent(N)
# print(selected_numbers)

# def test():
#     print("L")
    
# def test1():
#     print("M")

# from itertools import combinations

# def find_subsets(nums, ratio):
#     n = len(nums)
#     subsets = []
    
#     # calculate the range of the subset lengths based on the ratio
#     min_len = int(n * ratio)
#     max_len = int(n / ratio)
    
#     for r in range(min_len, max_len + 1):
#         for subset in combinations(nums, r):
#             subsets.append(list(subset))
    
#     return subsets

# nums = [0,1,2,3,4]
# ratio = 0.2
# print(find_subsets(nums, ratio))  # Outputs: [[0], [1], [2], [3], [4]]

# ratio = 0.4
# print(find_subsets(nums, ratio))  # Outputs: ... [0, 1], [0, 2], [0, 3], ... , [3, 4]


# from itertools import combinations

# def generate_combinations(elements, ratio):
#     n = len(elements)
#     max_len = int((n+1) * ratio)
    
#     result = []
#     for r in range(1, max_len + 1):
#         result.extend(combinations(elements, r))
    
#     return [list(comb) for comb in result]

# lst = [0, 1, 2, 3, 4]
# ratio1 = 0.2
# ratio2 = 0.4

# print(generate_combinations(lst, ratio1))
# print(generate_combinations(lst, ratio2))

# def integrate_quantification(value1, value2, value3, weight1, weight2, weight3):
#     # 首先确保权重的总和为1，以保证加权后的结果仍然在[0,1]范围内
#     total_weight = weight1 + weight2 + weight3
#     if total_weight != 1:
#         raise ValueError("The sum of weights must be 1.")
    
#     # 计算加权总和
#     integrated_value = value1 * weight1 + value2 * weight2 + value3 * weight3
    
#     return integrated_value

# # 示例
# v1 = 0.5
# v2 = 0.7
# v3 = 0.2
# w1 = 0.4
# w2 = 0.4
# w3 = 0.2

# print(integrate_quantification(v1, v2, v3, w1, w2, w3))  # 0.52

import json

from classes.Nodes import Nodes
from functions.generate_users import generate_users
from functions.generate_UAVs import generate_UAVs
from functions.calculate_data_rate import calculate_data_rate
from functions.dB_conversion import dB_conversion
from functions.data_unit_conversion import data_unit_conversion
from classes.UAVMap import UAVMap
from functions.path_is_blocked import path_is_blocked
from functions.quantify_data_rate import quantify_data_rate
from functions.quantify_backup_path import quantify_backup_path
from functions.quantify_network_partitioning import quantify_network_partitioning
from functions.integrate_quantification import integrate_quantification
from functions.measure_overload import measure_overload
from functions.print_nodes import print_nodes


from functions.print_nodes import print_nodes
from classes.UAVMap import UAVMap
from functions.quantify_data_rate import quantify_data_rate
from functions.quantify_backup_path import quantify_backup_path
from functions.quantify_network_partitioning import quantify_network_partitioning
from functions.integrate_quantification import integrate_quantification

import copy

import numpy as np
import random
from collections import deque
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.train_on_batch(state, target_f)
            # self.model.train_on_batch(state, target_f, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class UAVEnvironment:
    def __init__(self, UAVNodes, ABSNodes, blocks, UAVInfo):
        # self.UAVNodes = [
        #     (97.98413041207571, 663.8191957481255, 200),
        #     (335.5433790862149, 84.90963373333902, 200),
        #     (200.30808150599717, 565.5323906186599, 200),
        #     (79.47903810469664, 332.2894488804361, 200),
        #     (383.1300553845101, 141.8551700103783, 200)
        # ]
        
        self.UAVNodes = copy.deepcopy(UAVNodes)
        self.ABSNodes = copy.deepcopy(ABSNodes)
        self.blocks = copy.deepcopy(blocks)
        self.UAVInfo = copy.deepcopy(UAVInfo)
        
        self.UAVNodesReset = copy.deepcopy(UAVNodes)
        
        # print("Test")
        # print(UAVNodes)
        
        # UAVNodes[0].set_position((2,3,4))
        # print_nodes(UAVNodes, onlyPosition=True)
        # print_nodes(ABSNodes, onlyPosition=True)
        # print_nodes(blocks)
        # print(UAVInfo)
        
        self.UAVMap = UAVMap(UAVNodes, ABSNodes, blocks, UAVInfo)
        print(self.UAVMap)
        
        self.get_RS()
        

    def set_position(self, node_index, new_position):
        # self.UAVNodes[node_index] = position
        self.UAVNodes[node_index].set_position(new_position)

    def get_RS(self):
        # # Dummy function, replace with actual RS computation
        # return np.random.rand()
        
        # quantify resilience score: data rate
        DRPenalty = 0.5
        DRScore = quantify_data_rate(self.UAVMap, DRPenalty, self.UAVInfo)
        # print("Data rate score of current topology:")
        # print(DRScore)

        # for node in UAVNodes:
        #     print(node)

        # quantify resilience score: backup path 
        BPHopConstraint = 4
        BPDRConstraint = 100000000
        BPScore = quantify_backup_path(self.UAVMap, BPHopConstraint, BPDRConstraint)
        # print("Backup path score of current topology:")
        # print(BPScore)

        # quantify resilience score: network partitioning
        # for node in UAVNodes:
        #     print(node)
        # newUAVMap = remove_node(UAVMap, 2)
        # print(newUAVMap)

        droppedRatio = 0.2
        ratioDR = 0.6
        ratioBP = 0.4
        NPScore = quantify_network_partitioning(self.UAVMap, droppedRatio, DRPenalty, BPHopConstraint, BPDRConstraint,self.UAVInfo, DRScore, BPScore, ratioDR, ratioBP)
        # print("Network partitioning score of current topology:")
        # print(NPScore)

        # integrate quantificaiton
        weightDR = 0.2
        weightBP = 0.5
        weightNP = 0.3
        ResilienceScore = integrate_quantification(DRScore, BPScore, NPScore, weightDR, weightBP, weightNP)
        print("Resilience score is:")
        print(ResilienceScore) 
        
        return ResilienceScore

    def reset(self):
        # Reset UAV positions to initial state
        # self.UAVNodes = [
        #     (97.98413041207571, 663.8191957481255, 200),
        #     (335.5433790862149, 84.90963373333902, 200),
        #     (200.30808150599717, 565.5323906186599, 200),
        #     (79.47903810469664, 332.2894488804361, 200),
        #     (383.1300553845101, 141.8551700103783, 200)
        # ]
        # return np.array(self.UAVNodes).flatten()

        # self.UAVNodes = copy.deepcopy(self.UAVNodesReset)
        # # for node in self.UAVNodes:
        # #     print(node.position)
        # return self
        # return  np.array(self.UAVNodes.position).flatten()
        
        self.UAVNodes = copy.deepcopy(self.UAVNodesReset)
        
        state = [node.position for node in self.UAVNodes]
        return np.array(state).flatten()


    
    def step(self, action):
        # define action and action spaces
        uav_index = action // 5  # consider no need to change the aptitude, there are 5 steps
        uav_action = action % 5

        # Adjust UAV's position based on actions
        x, y, z = self.UAVNodes[uav_index].position
        if uav_action == 0:  # forward
            y += 1
        elif uav_action == 1:  # downward
            y -= 1
        elif uav_action == 2:  # leftward
            x -= 1
        elif uav_action == 3:  # rightward
            x += 1
        # if uav_action == 4，stay and dont move

        # since UAV only move in the same aptitude, z does not to be modified
        self.UAVNodes[uav_index].set_position((x, y, z))

        # calculate reward (RS based on new UAV position)
        self.UAVMap = UAVMap(self.UAVNodes, self.ABSNodes, self.blocks, self.UAVInfo)
        reward = self.get_RS()

        # check whether it is finished
        # we can judge whether we reach maximum step or other condition
        done = False

        # return new state, reward, and finish sign
        next_state = np.array([node.position for node in self.UAVNodes]).flatten()
        return next_state, reward, done


if __name__ == "__main__":
    
    with open('scene_data.json', 'r') as f:
        ini = json.load(f)

    groundBaseStation = ini['baseStation']
    blocks = ini['blocks']
    UAVInfo = ini['UAV']
    scene = ini['scenario']

    
    # Generate ground users
    ground_users = generate_users(5, blocks, scene['xLength'], scene['yLength'])

    # for user in ground_users:
    #     print(user)

    # Generate random UAVs
    defaultHeightUAV = 200
    UAVNodes = generate_UAVs(5, blocks, scene['xLength'], scene['yLength'], defaultHeightUAV, 10, 'basic UAV')
    # for node in UAVNodes:
    #     print(node)


    # Generate air base station
    defaultHeightABS = 500
    ABSNodes = generate_UAVs(2, blocks, scene['xLength'], scene['yLength'], defaultHeightABS, 10, 'Air Base Station')

    UAVNodes[0].set_connection([1,2,3])
    UAVNodes[1].set_connection([0])
    UAVNodes[2].set_connection([0])
    UAVNodes[3].set_connection([0])

    ABSNodes[0].set_connection([3])
    ABSNodes[1].set_connection([2,4])

    UAVMap1 = UAVMap(UAVNodes, ABSNodes, blocks, UAVInfo)
    
    env = UAVEnvironment(UAVNodes, ABSNodes, blocks, UAVInfo)
    
    state_size = len(env.UAVNodes) * 3
    action_size = len(env.UAVNodes) * 5  # For each UAV: move in x, move in y, or don't move
    agent = DQNAgent(state_size, action_size)
    episodes = 10
    
    rs_values = []

    # print(env)
    # state = env.reset()
    # state = np.reshape(state, [1, state_size])
    # print(state)
        
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            rs_current = env.get_RS()
            rs_values.append(rs_current)
            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
        if len(agent.memory) > 32:
            agent.replay(32)
            
    plt.plot(rs_values)
    plt.xlabel('Episode')
    plt.ylabel('RS Value')
    plt.title('RS Value over Episodes')
    plt.show()


