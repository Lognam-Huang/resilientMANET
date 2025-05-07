import gym
import numpy as np
from gym import spaces

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functions.calculate_data_rate import calculate_data_rate  # Use this for data rate calculations

class UAVEnvironment(gym.Env):
    def __init__(self, scene_data):
        super(UAVEnvironment, self).__init__()
        
        # Load scene data
        self.scene_data = scene_data
        self.ground_users = self._generate_ground_users(scene_data['nodeNumber']['GU'], scene_data['scenario'])
        self.uav_count = scene_data['nodeNumber']['UAV']
        self.height_range = [scene_data['UAV']['min_height'], scene_data['UAV']['max_height']]
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.tile([0, 0, self.height_range[0]], self.uav_count),  # Min values for all UAVs
            high=np.tile([scene_data['scenario']['xLength'], scene_data['scenario']['yLength'], self.height_range[1]], self.uav_count),  # Max values for all UAVs
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(self.ground_users),),  # Example: LOS status for each GU
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.uav_positions = None

    def _generate_ground_users(self, gu_count, scenario):
        """
        Generate random positions for ground users within the scenario bounds.
        """
        x_length = scenario['xLength']
        y_length = scenario['yLength']
        ground_users = []
        for _ in range(gu_count):
            x = np.random.uniform(0, x_length)
            y = np.random.uniform(0, y_length)
            z = 0  # Ground users are at height 0
            ground_users.append({'position': np.array([x, y, z])})
        return ground_users

    def reset(self):
        # Reset UAV positions and state
        self.uav_positions = np.zeros((self.uav_count, 3))  # Initialize UAV positions
        self.state = np.zeros(len(self.ground_users))  # Example: LOS status for each GU
        return self.state

    def step(self, action):
        # Update UAV positions based on action
        self.uav_positions = action.reshape(self.uav_count, 3)
        
        # Wrap UAV positions in objects with a `.position` attribute
        class Node:
            def __init__(self, position):
                self.position = position

        uav_objects = [Node(pos) for pos in self.uav_positions]
        
        # Calculate LOS and rewards
        los_count = 0
        for i, gu in enumerate(self.ground_users):
            gu_object = Node(gu['position'])  # Wrap GU position in an object
            los = any(not self.check_path_is_blocked(self.scene_data['blocks'], uav.position, gu_object.position) for uav in uav_objects)
            self.state[i] = 1 if los else 0
            if los:
                los_count += 1
        
        # Reward: Maximize LOS and balance GU assignments
        # reward = los_count - self._calculate_imbalance_penalty()

        # model performance is low, try to improve it by modifying the reward function
        # considerting los_count only, discard the imbalance penalty
        reward = los_count  # Example: Reward is the number of GUs covered
        
        # Check if the episode is done
        done = los_count == len(self.ground_users)  # Example: All GUs covered
        
        return self.state, reward, done, {}

    def render(self, mode='human'):
        # Optional: Visualize UAV positions and GU coverage
        pass

    def _calculate_imbalance_penalty(self):
        # Example: Penalize imbalance in GU assignments
        gu_assignments = [0] * self.uav_count
        for i, gu in enumerate(self.ground_users):
            if self.state[i] == 1:  # GU is covered
                closest_uav = np.argmin(np.linalg.norm(self.uav_positions - gu['position'], axis=1))
                gu_assignments[closest_uav] += 1
        imbalance = np.std(gu_assignments)  # Standard deviation as imbalance metric
        return imbalance

    def check_path_is_blocked(self, blocks, position_a, position_b):
        """
        Check if the path between two points (position_a and position_b) is blocked by any obstacles.

        Args:
            blocks (list): List of obstacles, each defined by a dictionary with 'bottomCorner', 'size', and 'height'.
            position_a (numpy.ndarray): Starting position (e.g., UAV position).
            position_b (numpy.ndarray): Ending position (e.g., GU position).

        Returns:
            bool: True if the path is blocked, False otherwise.
        """
        for block in blocks:
            # Extract block properties
            bottom_corner = np.array(block.get('bottomCorner', [0, 0, 0]))
            size = np.array(block.get('size', [0, 0, 0]))
            height = block.get('height', 0)

            # Define the block's bounding box
            block_min = bottom_corner
            block_max = bottom_corner + np.array([size[0], size[1], height])

            # Check if the line segment intersects the block
            if self.line_intersects_box(position_a, position_b, block_min, block_max):
                return True  # Path is blocked

        return False  # Path is not blocked

    def line_intersects_box(self, p1, p2, box_min, box_max):
        """
        Check if a line segment (p1 to p2) intersects an axis-aligned bounding box (box_min to box_max).

        Args:
            p1 (numpy.ndarray): Starting point of the line segment.
            p2 (numpy.ndarray): Ending point of the line segment.
            box_min (numpy.ndarray): Minimum corner of the bounding box.
            box_max (numpy.ndarray): Maximum corner of the bounding box.

        Returns:
            bool: True if the line intersects the box, False otherwise.
        """
        direction = p2 - p1
        tmin = (box_min - p1) / np.where(direction != 0, direction, np.inf)
        tmax = (box_max - p1) / np.where(direction != 0, direction, np.inf)

        t1 = np.minimum(tmin, tmax)
        t2 = np.maximum(tmin, tmax)

        t_enter = np.max(t1)
        t_exit = np.min(t2)

        return t_enter <= t_exit and t_exit >= 0