# mappo_example.py

import gym
import gym.spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 创建或加载一个多智能体环境
# 注意：你需要一个支持多智能体的环境，这里只是一个占位符
env = make_vec_env('YourMultiAgentEnv-v0', n_envs=4)

# 创建一个MAPPO模型
# 注意：MAPPO是PPO的一个扩展，可能需要自定义实现或查找支持多智能体的PPO版本
model = PPO('MlpPolicy', env, verbose=1)

# 训练模型
model.learn(total_timesteps=10000)

# 保存模型
model.save("mappo_model")

# 测试训练好的模型
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()

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


