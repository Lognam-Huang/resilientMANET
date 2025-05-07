import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from custom_uav_env import UAVEnvironment  # Custom environment for UAV positioning
import json

# Create output folder
output_folder = os.path.dirname(__file__)
date_str = datetime.now().strftime("%Y-%m-%d")
model_name = f"DRL_model_{date_str}"
model_path = os.path.join(output_folder, model_name)
data_path = os.path.join(output_folder, "training_data")
os.makedirs(data_path, exist_ok=True)

# Load scene data
scene_data_file = os.path.join(output_folder, "../scene_data_mid.json")
with open(scene_data_file, 'r') as f:
    scene_data = json.load(f)

# Initialize the custom environment
env = UAVEnvironment(scene_data)

# Wrap the environment for vectorized training
env = DummyVecEnv([lambda: env])

# Initialize the DRL model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
total_timesteps = 100000
print(f"Training the model for {total_timesteps} timesteps...")
training_rewards = []
obs = env.reset()

for i in range(total_timesteps):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    training_rewards.append(rewards)
    if done:
        obs = env.reset()

# Save the trained model
model.save(model_path)
print(f"Model saved as {model_name}")

# Save training rewards
training_rewards_file = os.path.join(data_path, "training_rewards.npy")
np.save(training_rewards_file, training_rewards)
print(f"Training rewards saved at {training_rewards_file}")

# Plot training rewards
plt.figure(figsize=(10, 6))
plt.plot(training_rewards, label="Training Rewards")
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title("Training Rewards Over Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(data_path, "training_rewards_plot.png"))
plt.show()

# Test the trained model
print("Testing the trained model...")
obs = env.reset()
test_rewards = []
for _ in range(1000):  # Test for 1000 steps
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    test_rewards.append(rewards)
    if done:
        obs = env.reset()

# Save testing rewards
test_rewards_file = os.path.join(data_path, "test_rewards.npy")
np.save(test_rewards_file, test_rewards)
print(f"Test rewards saved at {test_rewards_file}")

# Plot testing rewards
plt.figure(figsize=(10, 6))
plt.plot(test_rewards, label="Test Rewards")
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title("Test Rewards Over Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(data_path, "test_rewards_plot.png"))
plt.show()