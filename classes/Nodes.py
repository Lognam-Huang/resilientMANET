import numpy as np

class Nodes:
    def __init__(self, position=(0, 0, 0), node_type="NA", data_rate=0, node_number=0):
        """
        Initialize a new node with the given parameters.

        Parameters:
        position (tuple): The (x, y, z) coordinates of the node.
        node_type (str): The type of the node (e.g., "ground users", "basic UAV", "Air Base Station").
        data_rate (float): The data rate of the node.
        node_number (int): The unique identifier number of the node.
        """
        self.position = position
        self.type = node_type
        self.data_rate = data_rate
        self.connected_nodes = []
        self.node_number = node_number

    def __str__(self):
        """
        Return a string representation of the node's information, starting with a separator line.
        """
        separator = "=" * 30
        info = [
            separator,
            f"Node Information:",
            f"Position: {self.position}",
            f"Type: {self.type}",
            f"Data Rate: {self.data_rate}",
            f"Node Number: {self.node_number}"
        ]

        type_mapping = {
            "ground users": "UAVs connected to current ground users are:",
            "basic UAV": "UAVs connected to current UAV are:",
            "Air Base Station": "UAV connected to current ABS are:"
        }
        
        info.append(type_mapping.get(self.type, "Unknown type of Nodes"))

        if self.connected_nodes:
            connected_info = [f"Node {node}, " for node in self.connected_nodes]
            info.append(''.join(connected_info).rstrip(', '))
        else:
            info.append("None")

        return '\n'.join(info)

    def set_position(self, new_position):
        """
        Set the position of the node to the new position if it's a tuple or numpy array of length 3.
        """
        if isinstance(new_position, (tuple, np.ndarray)) and len(new_position) == 3:
            self.position = tuple(new_position)
        else:
            print("Error: Please enter a valid position in the format (x, y, z).")
    
    def set_connection(self, cell_data):
        """
        Set the connected nodes based on the input.
        If the input is an integer or a list of integers, replace the current connected nodes with it.

        Parameters:
        cell_data (int or list): A single integer or a list of integers representing node numbers.
        """
        if isinstance(cell_data, int) and cell_data >= 0:
            self.connected_nodes = [cell_data]
        elif isinstance(cell_data, list) and all(isinstance(item, int) and item >= 0 for item in cell_data):
            self.connected_nodes = cell_data
        else:
            raise ValueError("Input must be a non-negative integer or a list of non-negative integers.")

    def add_connection(self, new_connection):
        """
        Add a single connection or multiple connections to the node.

        Parameters:
        new_connection (int or list): The new connection(s) to add. Can be a single integer or a list of integers.
        """
        if isinstance(new_connection, int):
            if new_connection >= 0:
                self.connected_nodes.append(new_connection)
            else:
                print("Illegal node index: Index must be non-negative.")
        elif isinstance(new_connection, list):
            for item in new_connection:
                if isinstance(item, int) and item >= 0:
                    self.connected_nodes.append(item)
                else:
                    print(f"Illegal value: {item}. Index must be a non-negative integer.")
        else:
            print("Illegal input: Input must be an integer or a list of integers.")



