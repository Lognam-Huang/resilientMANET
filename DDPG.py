# Import necessary libraries 
import numpy as np 
import random 
import tensorflow as tf 

from functions.print_nodes import print_nodes
from classes.UAVMap import UAVMap
from functions.quantify_data_rate import quantify_data_rate
from functions.quantify_backup_path import quantify_backup_path
from functions.quantify_network_partitioning import quantify_network_partitioning
from functions.integrate_quantification import integrate_quantification

import copy
# Define the environment class
class Environment:
    def __init__(self, UAVNodes, ABSNodes, blocks, UAVInfo):
        self.UAVNodes = copy.deepcopy(UAVNodes)
        self.ABSNodes = copy.deepcopy(ABSNodes)
        self.blocks = copy.deepcopy(blocks)
        self.UAVInfo = copy.deepcopy(UAVInfo)
        
        self.UAVMap = UAVMap(UAVNodes, ABSNodes, blocks, UAVInfo)
        print(self.UAVMap)
    
    def calculate_RS(self): 
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
    
    def step(self, actions): # Update the positions of UAVNodes based on actions 
        for i, action in enumerate(actions): 
            new_position = [self.UAVNodes[i].position[0] + action[0], self.UAVNodes[i].position[1] + action[1], 0] 
            self.UAVNodes[i].set_position(new_position) 
            
        # Calculate the new RS 
        rs = self.calculate_RS() 
        return rs 
    
    def reset(self):
        # Reset UAV positions to initial state
        self.UAVNodes = copy.deepcopy(self.UAVNodesReset)
        state = [node.position for node in self.UAVNodes]
        return np.array(state).flatten()
    
# Define the DDPG agent class 
class DDPGAgent: 
    def __init__(self, state_dim, action_dim): 
        self.state_dim = state_dim 
        self.action_dim = action_dim 
        self.actor_model = self.create_actor_model() 
        self.critic_model = self.create_critic_model() 
        self.target_actor_model = self.create_actor_model() 
        self.target_critic_model = self.create_critic_model() 
        self.target_actor_model.set_weights(self.actor_model.get_weights()) 
        self.target_critic_model.set_weights(self.critic_model.get_weights()) 
    
    def create_actor_model(self):
        model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(self.state_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(self.action_dim, activation='linear')
    ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def create_critic_model(self):
        state_input = tf.keras.Input(shape=(self.state_dim,))
        action_input = tf.keras.Input(shape=(self.action_dim,))
        concat = tf.keras.layers.Concatenate()([state_input, action_input])
        hidden1 = tf.keras.layers.Dense(128, activation='relu')(concat)
        hidden2 = tf.keras.layers.Dense(64, activation='relu')(hidden1)
        output = tf.keras.layers.Dense(1, activation='linear')(hidden2)
        model = tf.keras.Model([state_input, action_input], output)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

# Initialize environment and agent
env = Environment()
state_dim = 10  # Placeholder, you would define the actual state dimension based on your specific problem
action_dim = 2  # Since UAVs can move in x and y directions
agent = DDPGAgent(state_dim, action_dim)

# Placeholder for DDPG training loop
# In a real-world scenario, you would implement the actual training loop here

print("Initialization complete. Ready for DDPG training.")