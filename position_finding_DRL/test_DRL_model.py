import os
import numpy as np
import json
from stable_baselines3 import PPO
from DRL_uav_env import UAVEnvironment

# Load the trained model
output_folder = os.path.dirname(__file__)
# model_name = "DRL_model_2025-05-07"  # Replace with the actual model name if different
# model_name = "DRL_model_2025-05-07_12-55-15"  

# try model trained in simple scene
# model_name = "DRL_model_2025-05-08_18-45-05" 
model_name = "DRL_model_2025-05-08_19-31-15" 

model_path = os.path.join(output_folder, model_name)
model = PPO.load(model_path)

# Load scene data
# 

# try simple scene data
scene_data_file = os.path.join(output_folder, "../scene_data_simple.json")
with open(scene_data_file, 'r') as f:
    scene_data = json.load(f)

# Initialize the environment
env = UAVEnvironment(scene_data)

# # Set GU positions explicitly (same as in the JSON file)
# gu_positions = [
#     [10, 170, 0], [40, 40, 0], [60, 150, 0], [90, 100, 0],
#     [120, 40, 0], [140, 160, 0], [160, 70, 0], [180, 10, 0],
#     [50, 50, 0], [70, 70, 0], [110, 110, 0], [150, 150, 0]
# ]

# set GU positions in simple scene, time 0
gu_positions = [
    [16.639898466459577, 6.810175094661559, 0],
    [23.459180130034913, 6.08773126277444, 0],
    [13.959450921539249, 24.14646634485348, 0],
    [3.340772507821935, 0.5020684860183979, 0],
    [9.926102492064935, 0.8290653849936879, 0],
    [10.54463978446006, 5.902074649607489, 0]
]

env.ground_users = [{'position': np.array(pos)} for pos in gu_positions]

# Reset the environment
obs = env.reset()

# Use the model to predict UAV positions
action, _ = model.predict(obs)

# Reshape the action to get UAV positions
uav_positions = action.reshape(env.uav_count, 3)

# Print the UAV positions
print("Predicted UAV Positions:")
for i, pos in enumerate(uav_positions):
    print(f"UAV {i + 1}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")