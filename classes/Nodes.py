class Nodes:
    def __init__(self, position=[0, 0, 0], node_type="NA", data_rate=0):
        self.position = position
        self.type = node_type
        self.data_rate = data_rate
        self.connected_nodes = []

    def __str__(self):
        info = [
            f"Node Information:",
            f"Position: {self.position}",
            f"Type: {self.type}",
            f"Data Rate: {self.data_rate}"
        ]
        
        if self.type == "ground users":
            info.append("UAVs connected to current ground users are: ")
        elif self.type == "basic UAV":
            info.append("UAVs connected to current UAV are:")
        elif self.type == "Air Base Station":
            info.append("UAV connected to current ABS are:")
        else:
            info.append("Unknown type of Nodes")

        if not self.connected_nodes:
            info.append("None")
        else:
            connected_info = ["Node {}, ".format(node) for node in self.connected_nodes]
            info.append(''.join(connected_info))

        return '\n'.join(info)

    def set_position(self, new_position):
        # Validate the input format
        if isinstance(new_position, tuple) and len(new_position) == 3 and all(isinstance(coord, (int, float)) for coord in new_position):
            self.position = new_position
            # print("successfully change position")
        else:
            print("Error: Please enter a valid position in the format (x, y, z).")


    def set_connection(self, cell_data):
        if isinstance(cell_data, list):
            self.connected_nodes = cell_data
        else:
            raise ValueError("Input must be a list.")
        
    def add_connection(self, new_connection):
        if isinstance(new_connection, int):
            if new_connection < 0:
                print("Illegal node index")
            else:                
                self.connected_nodes.append(new_connection)
                
        elif isinstance(new_connection, list):
            for item in new_connection:
                if not isinstance(item, int):
                    print("Index not integer")
                if item < 0:
                    print("Illegal node index")
                else:
                    self.connected_nodes.append(new_connection)
        else:
            print("Illegal input")


