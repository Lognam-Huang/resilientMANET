def print_nodes(all_nodes, position = False):
    for node in all_nodes:
        if position == True:
            print(node.position)
        else:
            print(node)