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
        Generate random positions for ground users within the scenario bounds,
        ensuring no collisions with blocks.
        """
        x_length = scenario['xLength']
        y_length = scenario['yLength']
        blocks = self.scene_data['blocks']
        soft_margin = 1.0  # Margin to avoid placing GUs too close to block edges
        ground_users = []

        for _ in range(gu_count):
            while True:
                # Generate random x and y positions
                x = np.random.uniform(0, x_length)
                y = np.random.uniform(0, y_length)
                z = 0  # Ground users are at height 0

                # Check for collisions with blocks
                collision = False
                for block in blocks:
                    bx, by, _ = block['bottomCorner']
                    bw, bh = block['size']
                    if (bx - soft_margin <= x <= bx + bw + soft_margin and
                        by - soft_margin <= y <= by + bh + soft_margin):
                        collision = True
                        break

                # If no collision, add the GU position and break the loop
                if not collision:
                    ground_users.append({'position': np.array([x, y, z])})
                    break

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
        
        # Calculate LOS and assign GUs to UAVs
        los_count = 0
        gu_assignments = [0] * self.uav_count  # Track the number of GUs assigned to each UAV
        for i, gu in enumerate(self.ground_users):
            gu_object = Node(gu['position'])  # Wrap GU position in an object
            best_uav = None
            best_distance = float('inf')
            for uav_index, uav in enumerate(uav_objects):
                if not self.check_path_is_blocked(self.scene_data['blocks'], uav.position, gu_object.position):
                    distance = np.linalg.norm(uav.position - gu_object.position)
                    if distance < best_distance:
                        best_distance = distance
                        best_uav = uav_index
            if best_uav is not None:
                los_count += 1
                gu_assignments[best_uav] += 1

        # Calculate imbalance penalty
        max_assignments = max(gu_assignments)
        min_assignments = min(gu_assignments)
        imbalance_penalty = max_assignments - min_assignments

        # Reward: Maximize LOS and minimize imbalance
        # try to encourage UAVs to be evenly distributed
        reward = los_count*10 - imbalance_penalty + 0.1 * np.mean(np.linalg.norm(self.uav_positions, axis=1))

        # Print reward and relevant positions
        # self.print_reward_info(reward, los_count, imbalance_penalty, gu_assignments, self.uav_positions, self.ground_users)

        # Check if the episode is done
        done = los_count == len(self.ground_users)  # Example: All GUs covered
        
        return self.state, reward, done, {}

    def print_reward_info(self, reward, los_count, imbalance_penalty, gu_assignments, uav_positions, ground_users):
        """
        Print the reward and relevant positions (GU and UAV) for calculating the reward.
        """
        print("\n" + "-" * 50)
        print(f"Reward: {reward}")
        print(f"LOS Count: {los_count}")
        print(f"Imbalance Penalty: {imbalance_penalty}")
        print(f"GU Assignments per UAV: {gu_assignments}")
        print("UAV Positions:")
        for i, pos in enumerate(uav_positions):
            print(f"  UAV {i + 1}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
        print("Ground User Positions:")
        for i, gu in enumerate(ground_users):
            pos = gu['position']
            print(f"  GU {i + 1}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
        print("-" * 50 + "\n")

    def render(self, mode='human'):
        # Optional: Visualize UAV positions and GU coverage
        pass

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