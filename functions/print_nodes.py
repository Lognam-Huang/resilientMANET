def print_nodes(all_nodes, onlyPosition = False):
    for node in all_nodes:
        if onlyPosition == True:
            print(node.position)
        else:
            print(node)

def print_specific_nodes(all_nodes, node_number ,onlyPosition = False):
    for node in all_nodes:
        if node.node_number == node_number:
            if onlyPosition == True:
                print(node.position)
            else:
                print(node)

def print_node_number(node):
    print(node.node_number)

def get_nodes_position(all_nodes):
    positions = []
    for node in all_nodes:
        positions.append(node.position)
    return positions
