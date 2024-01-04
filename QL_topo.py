import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# creating q table
import pandas as pd
from itertools import combinations

# quantify RS and overload
from quantify_topo import *

# 节点坐标
node_coords = np.array([
    (10, 10, 10),
    (20, 30, 10),
    (40, 100, 15),
    (10, 90, 15)
])

UAV_coords = np.array([
    (250,200,200),
    (250,600,200),
    (600,350,200),
])

ABS_coords = np.array([
    (440,390,500),
])



# num_nodes = len(node_coords)
num_nodes = len(ABS_coords) + len(UAV_coords)

actions = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)]
# Q = defaultdict(lambda: np.zeros(len(actions)))  # 初始化Q表

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

# Example usage:
# graph_table = create_graph_table(3)
# print(graph_table)

# Q-learning 参数
# hyperparameters
epsilon = 0.1
# randomness of choosing actions
alpha = 0.5
# learning rate
gamma = 0.9
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
    print("Resilience score is:")
    print(ResilienceScore) 

    # as for the reward function, we need also to consider the balance in the UAV network
    # here we use gini coefficient
    overloadConstraint = 10000
    OverloadScore = measure_overload(UAVMap, BPHopConstraint, BPDRConstraint, overloadConstraint)
    print("Overload score is:")
    print(OverloadScore)

    # now we just return RS*overload
    return ResilienceScore*OverloadScore


# def take_action(state, action, add=True):
#     new_state = state.copy()
#     if add:
#         new_state[action] = 1  # 添加连接
#     else:
#         new_state[action] = 0  # 移除连接
#     reward = Reward(new_state) - Reward(state)
#     return new_state, reward


def take_action(state, epsilon, q_table):
    # check if current state is initialized or not
    # randomly choose action:
    #     perform a legal action
    #     perform the action with the highest reward in the q_table
    
    # after choosing the action, generate the new state based on current state and action
    # update reward --> his is put in training process
    print("Bye")

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
        # cur_action = q_table.loc[state].idxmax()
        # cur_value = feasible_actions[cur_action]
        cur_action = q_table.loc[state].idxmax()
        cur_value = q_table.loc[state, cur_action]
        
        
        
    new_state = get_new_state(state, cur_action)
    initialize_state(new_state, q_table)
    return new_state, cur_action, cur_value



def initialize_state(state, q_table):
    print("hi")
    # for a new table, if a state does not contain any meaningful data, should create some records

    # Convert state to row index
    row_index = state

    print(q_table)
    # print(q_table.loc[row_index])
    for value in q_table.loc[row_index]:
        print(value)

    
    # Check if the state has already been initialized
    if not all(value in [0, -1] for value in q_table.loc[row_index]):
        # State has been initialized

        print("State has been initialized")
        return
    
    # Initialize the state row
    for col in q_table.columns:
        if q_table.at[row_index, col] == 0:  # Check if the cell needs to be updated

            print(state, col)
            print("row_index= " +row_index)

            # Compute the new state by applying the action
            new_state = get_new_state(state, col)
            # Get the reward for the new state
            reward = Reward(new_state)
            # Update the Q table

            # Notice that although we should use .at() to modify values in dataframe, it cannot work well here
            # instead of that, we also try to directly visit q_table, but it will not modify values---it will create new rows
            # so far, I notice that loc() seems to successfull modify values

            print("Current reward = "+str(reward))
            # q_table.at[row_index, col] = reward
            # q_table[row_index, col] = reward
            q_table.loc[row_index, col] = reward
            
    
    print(q_table)

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

# 存储每个episode的RS值
rs_values = []


# print(num_nodes)
# print(actions)
# print(Q)
# # print state
# state = np.zeros(len(actions))
# print(state)
# print(sum(state))

q_table = create_q_table(num_nodes)
# print(q_table)

# initialize_state("000000", q_table)
# initialize_state("000000", q_table)

# print(q_table.loc["000000"].idxmax())
# print(q_table.loc["000000"][q_table.loc["000000"] != -1])
# print(random.choice(q_table.loc["000000"][q_table.loc["000000"] != -1]))





# state[1] = 1
# state[2] = 1

# state[4] = 1
# state[5] = 1

# aa()


# state[3] = 1
# print(state)

# print(np.random.choice(len(actions)))


# Q-learning
for episode in range(1000):
    state = np.zeros(len(actions))  # 初始状态

    while True:  # 定义一个终止条件
        # # 选择行动
        # if np.random.rand() < epsilon:
        #     action = np.random.choice(len(actions))
        # else:
        #     action = np.argmax(Q[str(state)])

        # execute action
        new_state, action, reward = take_action(state, epsilon, q_table)

        # update q_table
        next_best_action = q_table.loc[new_state].idxmax()
        next_best_value = q_table.loc[new_state, next_best_action]
        td_target = reward + gamma * next_best_value
        td_delta = td_target - reward
        q_table.loc[state, action] += alpha * td_delta
        
        # # 更新Q表
        # best_next_action = np.argmax(Q[str(new_state)])
        # td_target = reward + gamma * Q[str(new_state)][best_next_action]
        # td_delta = td_target - Q[str(state)][action]
        # Q[str(state)][action] += alpha * td_delta

        # 更新状态
        state = new_state


        if sum(state)>=len(node_coords)*(len(node_coords)-1)/2:
            break

    # 记录RS值
    rs_values.append(Reward(state))

# 可视化RS值
plt.plot(rs_values)
plt.title('RS Value Over Episodes')
plt.xlabel('Episode')
plt.ylabel('RS Value')
plt.show()
