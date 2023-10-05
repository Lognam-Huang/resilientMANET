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

from functions.quantify_network_partitioning import remove_node, select_drop

# read scenario data
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
# for node in ABSNodes:
#     print(node)

fakeGroundUser = [100, 200, 0]
fakeUAV = [100, 200, 150]

# testDataRateNoBlocks = calculate_data_rate(UAVInfo['bandwidth'], dB_conversion(UAVInfo['Power'], 'dBm')/1000, UAVInfo['TransmittingAntennaGain'], UAVInfo['ReceievingAntennaGain'], UAVInfo['CarrierFrequency'], fakeUAV, fakeGroundUser, False)
testDataRateNoBlocks = calculate_data_rate(UAVInfo, fakeUAV, fakeGroundUser, False)
# print("No block")
[convertedDataRateNoBlocks, convertedUnitNoBlocks] = data_unit_conversion(testDataRateNoBlocks, 'bps')
# print(str(convertedDataRateNoBlocks)+" "+convertedUnitNoBlocks)

# testDataRateWithBlocks = calculate_data_rate(UAVInfo['bandwidth'], dB_conversion(UAVInfo['Power'], 'dBm')/1000, UAVInfo['TransmittingAntennaGain'], UAVInfo['ReceievingAntennaGain'], UAVInfo['CarrierFrequency'], fakeUAV, fakeGroundUser, True)
testDataRateWithBlocks = calculate_data_rate(UAVInfo, fakeUAV, fakeGroundUser, True)
# print("With block")
[convertedDataRateWithBlocks, convertedUnitWithBlocks] = data_unit_conversion(testDataRateWithBlocks, 'bps')
# print(str(convertedDataRateWithBlocks)+" "+convertedUnitWithBlocks)

# print(UAVNodes[0].position)

# UAVNodes[0].set_position((340,490,50))
# UAVNodes[1].set_position((440,390,50))

# print(UAVNodes[0].position)

# # print(UAVNodes[0].position)
# print(path_is_blocked(blocks, UAVNodes[0], UAVNodes[1]))

# UAVNodes[0].set_position((0,0,0))
# UAVNodes[1].set_position((500,500,500))

# # print(UAVNodes[0].position)
# print(path_is_blocked(blocks, UAVNodes[0], UAVNodes[1]))

# Find all available paths in MANET topology

# UAVNodes[0].set_position((2,3,4))
# UAVNodes[0].set_connection([1,5])
# print(UAVNodes[0])

# NOTICE THAT ALL NODES START WITH INDEX 0

UAVNodes[0].set_connection([1,2,3])
UAVNodes[1].set_connection([0])
UAVNodes[2].set_connection([0])
UAVNodes[3].set_connection([0])

ABSNodes[0].set_connection([3])
ABSNodes[1].set_connection([2,4])

UAVMap = UAVMap(UAVNodes, ABSNodes, blocks, UAVInfo)
print(UAVMap)

# for node in UAVNodes:
#     print(node)

# quantify resilience score: data rate
DRPenalty = 0.5
DRScore = quantify_data_rate(UAVMap, DRPenalty, UAVInfo)
print("Data rate score of current topology:")
print(DRScore)

# for node in UAVNodes:
#     print(node)

# quantify resilience score: backup path 
BPHopConstraint = 4
BPDRConstraint = 100000000
BPScore = quantify_backup_path(UAVMap, BPHopConstraint, BPDRConstraint)
print("Backup path score of current topology:")
print(BPScore)

# quantify resilience score: network partitioning
# for node in UAVNodes:
#     print(node)
# newUAVMap = remove_node(UAVMap, 2)
# print(newUAVMap)

droppedRatio = 0.2
ratioDR = 0.6
ratioBP = 0.4
NPScore = quantify_network_partitioning(UAVMap, droppedRatio, DRPenalty, BPHopConstraint, BPDRConstraint, UAVInfo, DRScore, BPScore, ratioDR, ratioBP)
print("Network partitioning score of current topology:")
print(NPScore)

# integrate quantificaiton
weightDR = 0.2
weightBP = 0.5
weightNP = 0.3
ResilienceScore = integrate_quantification(DRScore, BPScore, NPScore, weightDR, weightBP, weightNP)
print("Resilience score is:")
print(ResilienceScore) 

# measure the overload of current MANET topology
overloadConstraint = 10000
OverloadScore = measure_overload(UAVMap, BPHopConstraint, BPDRConstraint, overloadConstraint)
print(OverloadScore)

# try to use RL to improve resilience score
print_nodes(UAVNodes, position=True)


import numpy as np
import random
from collections import deque
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


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
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class UAVEnvironment:
    def __init__(self):
        self.UAVNodes = [
            (97.98413041207571, 663.8191957481255, 200),
            (335.5433790862149, 84.90963373333902, 200),
            (200.30808150599717, 565.5323906186599, 200),
            (79.47903810469664, 332.2894488804361, 200),
            (383.1300553845101, 141.8551700103783, 200)
        ]

    def set_position(self, node_index, position):
        self.UAVNodes[node_index] = position

    def get_RS(self):
        # Dummy function, replace with actual RS computation
        return np.random.rand()

    def reset(self):
        # Reset UAV positions to initial state
        self.UAVNodes = [
            (97.98413041207571, 663.8191957481255, 200),
            (335.5433790862149, 84.90963373333902, 200),
            (200.30808150599717, 565.5323906186599, 200),
            (79.47903810469664, 332.2894488804361, 200),
            (383.1300553845101, 141.8551700103783, 200)
        ]
        return np.array(self.UAVNodes).flatten()

    def step(self, action):
        # Dummy function, replace with actual logic to adjust UAV positions
        # and compute the resulting RS
        node_index = action // 3
        move_direction = action % 3
        if move_direction == 0:
            self.set_position(node_index, (self.UAVNodes[node_index][0] + 1, self.UAVNodes[node_index][1], 200))
        elif move_direction == 1:
            self.set_position(node_index, (self.UAVNodes[node_index][0], self.UAVNodes[node_index][1] + 1, 200))
        # else: no movement
        next_state = np.array(self.UAVNodes).flatten()
        reward = self.get_RS()
        done = False  # You can define a termination condition if needed
        return next_state, reward, done

if __name__ == "__main__":
    env = UAVEnvironment()
    state_size = len(env.UAVNodes) * 3
    action_size = len(env.UAVNodes) * 3  # For each UAV: move in x, move in y, or don't move
    agent = DQNAgent(state_size, action_size)
    episodes = 100

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
        if len(agent.memory) > 32:
            agent.replay(32)





# test
# print(data['scenario']['xLength'])  # 500
# print(data['baseStation']['bottomCorner'])  # [0, 0]

# print(groundBaseStation)
# print(blocks)
# print(UAVInfo)
# print(UAVInfo['bandwidth'])
# print(scene)

# Usage example
# node = Nodes([1, 2, 3], "basicUAV", 100)
# print(node)

# node.set_position([4, 5, 6])
# node.set_connection([1, 2, 3])
# print(node)

# testDrop = select_drop(UAVMap, 0.4)
# print(testDrop)

# print(UAVMap)
# print(newUAVMap)
# from test import test, test1
# test1()