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

        self.UAVNodes = copy.deepcopy(self.UAVNodesReset)
        state = [node.position for node in self.UAVNodes]
        return np.array(state).flatten()
        # return self
    
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
    env = UAVEnvironment()
    state_size = len(env.UAVNodes) * 3
    action_size = len(env.UAVNodes) * 5  # For each UAV: move in x, move in y, or don't move
    agent = DQNAgent(state_size, action_size)
    episodes = 10

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(100):
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

