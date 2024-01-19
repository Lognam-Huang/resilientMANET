import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# creating q table
import pandas as pd
from itertools import combinations

# quantify RS and overload
from quantify_topo import *

# node coordinations
node_coords = np.array([
    (10, 10, 10),
    (20, 30, 10),
    (40, 100, 15),
    (10, 90, 15)
])

# simple demonstration
UAV_coords = np.array([
    # (250,200,200),
    # (250,600,200),
    # (600,350,200),

    # (588, 127, 246),
    # (665, 310, 180),
    # (428, 777, 201),
    # (513, 769, 193),
    # (548, 317, 216),

    # (783, 626, 235),
    # (411, 254, 224),
    (600, 725, 224),
    (419, 38, 151),
    (423, 215, 183),
    # (643, 641, 198),
])

ABS_coords = np.array([
    # (440,390,500),

    # (294, 467, 500),
    (445, 0, 500),

    (511, 133, 500),
    # (244, 637, 500),
])




# num_nodes = len(node_coords)
num_nodes = len(ABS_coords) + len(UAV_coords)

actions = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)]

# Q table initialization
# for each row, it represents a state of connection, each elements represent an edge
# for each column, it represents an action of adding or deleting the edge of the connection

def create_q_table(n):
    # Generate all possible edges for a complete graph of n nodes
    edges = list(combinations(range(n), 2))

    # Generate row names: all possible states for edges (0 or 1 for each edge)
    num_edges = len(edges)
    row_names = [format(i, '0' + str(num_edges) + 'b') for i in range(2 ** num_edges)]

    # Generate column names: + and - for each edge
    col_names = ['+(' + str(i) + ',' + str(j) + ')' for i, j in edges] + \
                ['-(' + str(i) + ',' + str(j) + ')' for i, j in edges]

    # Create a DataFrame with 0s
    table = pd.DataFrame(0, index=row_names, columns=col_names)

    # Update the table to reflect the feasibility of each action in each state
    for row_name in row_names:
        # For each edge state in the row
        for i, edge_state in enumerate(row_name):
            # If the edge is 0 (not connected), '-' action is not feasible
            if edge_state == '0':
                table.at[row_name, '-(' + str(edges[i][0]) + ',' + str(edges[i][1]) + ')'] = -1
            # If the edge is 1 (connected), '+' action is not feasible
            else:
                table.at[row_name, '+(' + str(edges[i][0]) + ',' + str(edges[i][1]) + ')'] = -1

    return table


# Q-learning hyperparameters
epsilon = 0.1
# randomness of choosing actions
alpha = 0.2
# learning rate
gamma = 0.3
# discount rate

# Get reward of a state, including resilience score and optimization score
def Reward(state):
    # notice that score = RS-overload
    # or RS*overload

    UAVMap = get_UAVMap(state=state, UAV_position=UAV_coords, ABS_position=ABS_coords)

    # quantify resilience score: data rate
    DRPenalty = 0.5

    # quantify resilience score: backup path 
    BPHopConstraint = 4
    BPDRConstraint = 100000000

    # quantify resilience score: network partitioning
    droppedRatio = 0.2
    ratioDR = 0.6
    ratioBP = 0.4

    # integrate quantificaiton
    weightDR = 0.3
    weightBP = 0.4
    weightNP = 0.3

    ResilienceScore = get_RS(UAVMap, DRPenalty, BPHopConstraint, BPDRConstraint, droppedRatio, ratioDR, ratioBP, weightDR, weightBP, weightNP)
    # print("Resilience score is:")
    # print(ResilienceScore) 

    # as for the reward function, we need also to consider the balance in the UAV network
    # here we use gini coefficient
    overloadConstraint = 10000
    OverloadScore = measure_overload(UAVMap, BPHopConstraint, BPDRConstraint, overloadConstraint)
    # print("Overload score is:")
    # print(OverloadScore)

    # now we just return RS*overload
    rewardScore = ResilienceScore*OverloadScore

    # print("Reward score is:")
    # print(rewardScore)

    return rewardScore


def take_action(state, epsilon, q_table):
    # check if current state is initialized or not
    # randomly choose action:
    #     perform a legal action
    #     perform the action with the highest reward in the q_table
    
    # after choosing the action, generate the new state based on current state and action
    # update reward --> this is put in training process
    # print("take_action is executed")

    initialize_state(state, q_table)

    if np.random.rand() < epsilon:
        feasible_actions = q_table.loc[state][q_table.loc[state] != -1]
    
        # Randomly choose an action from the feasible actions
        if not feasible_actions.empty:
            cur_action = random.choice(feasible_actions.index)
            cur_value = q_table.loc[state, cur_action]
        else:
            cur_action = None
            cur_value = None
    else:
        cur_action = q_table.loc[state].idxmax()
        cur_value = q_table.loc[state, cur_action]
                  
    new_state = get_new_state(state, cur_action)
    initialize_state(new_state, q_table)

    # print("cur state: "+state)
    # print("next state: "+new_state)
    return new_state, cur_action, cur_value

def initialize_state(state, q_table):
    # print("initialize_state is executed")
    # for a new table, if a state does not contain any meaningful data, should create some records
    
    # Check if the state has already been initialized
    if not all(value in [0, -1] for value in q_table.loc[state]):
        # State has been initialized
        # print("State has been initialized")
        return
    
    # Initialize the state row
    for col in q_table.columns:
        if q_table.at[state, col] == 0:  # Check if the cell needs to be updated

            # Compute the new state by applying the action
            new_state = get_new_state(state, col)
            # Get the reward for the new state
            reward = Reward(new_state)
            # Update the Q table

            # Notice that although we should use .at() to modify values in dataframe, it cannot work well here
            # instead of that, we also try to directly visit q_table, but it will not modify values---it will create new rows
            # so far, I notice that loc() seems to successfull modify values

            # print("Initilize reward as:")
            # print(reward)
            q_table.loc[state, col] = reward
    
    # print(q_table)

def get_new_state(state, action):
    # Convert the state from string to list for easy manipulation
    state_list = list(state)
    
    # Parse the action to determine the type and edge
    action_type, edge_info = action[0], action[1:]  # '+' or '-', and the edge tuple '(i,j)'
    i, j = map(int, edge_info.strip('()').split(','))  # Extract edge indices from the action
    
    # Find the index of the edge in the state string
    n = int((1 + (1 + 8 * len(state))**0.5) / 2)  # Calculate n from the length of state string
    edges = list(combinations(range(n), 2))  # Generate the list of edges
    edge_index = edges.index((i, j))  # Get the index of the edge in the list
    
    # Update the state based on the action type
    if action_type == '+':  # Add edge
        state_list[edge_index] = '1' if state_list[edge_index] == '0' else state_list[edge_index]
    elif action_type == '-':  # Remove edge
        state_list[edge_index] = '0' if state_list[edge_index] == '1' else state_list[edge_index]
    
    # Convert the state list back to string and return
    return ''.join(state_list)

# store Reward for all episodes
reward_values = []
reward_track = []


q_table = create_q_table(num_nodes)
# print(q_table)

# this is used to set terminated case for Q-learning
# state_num: number of all available state
state_num = num_nodes*(num_nodes-1)/2

max_reward = 0

# Q-learning
# there are problems in creating state using np.zeros()
state = '0' * len(actions)

best_state = ""



# while True: 
for episode in range(10):
    while True:
        # choose and execute an action
        new_state, action, reward = take_action(state, epsilon, q_table)

        # print(new_state)
        # print(action)
        # print(reward)

        # update q_table
        next_best_action = q_table.loc[new_state].idxmax()
        next_best_value = q_table.loc[new_state, next_best_action]

        # this is the problem
        # td_target = reward + gamma * next_best_value
        td_target = reward + gamma * next_best_value - q_table.loc[state, action]

        # try to solve inf in q table
        td_delta = td_target - reward
        # td_delta = td_target - q_table.loc[state, action]

        # print(next_best_action)
        # print(next_best_value)


        q_table.loc[state, action] += alpha * td_delta

        # update state
        state = new_state

        if reward >= max_reward:
            max_reward = reward
            best_state = new_state

        # max_reward = max(max_reward, reward)

        reward_track.append(reward)

        # terminate condition
        # if the changes of q is too small, terminate the loop
        if td_delta < max(td_target, reward)*0.000001:
            print("this episode is terminated because the update is too small")
            break

        # print("the update is suitable")
        # print(state)
        state_sum = sum(int(char) for char in state)

        if state_sum>=state_num:
            print("this episode is terminated because current state is fully connected graph")
            break

# 记录RS值
# rs_values.append(Reward(state))
# print(max_reward)
# reward_values.append(max_reward)

print("Best RS value: "+str(max_reward))
print("Best topology: "+best_state)
# print(reward_values)
print(reward_track)
print(q_table)

# 可视化RS值
# plt.plot(reward_values)
plt.plot(reward_track)
plt.title('RS Value Over Episodes')
plt.xlabel('Episode')
plt.ylabel('RS Value')
plt.show()
