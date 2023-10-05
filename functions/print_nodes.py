def print_nodes(all_nodes, onlyPosition = False):
    for node in all_nodes:
        if onlyPosition == True:
            print(node.position)
        else:
            print(node)