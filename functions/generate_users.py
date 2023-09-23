import random

import sys
sys.path.append('../classes')
from classes.Nodes import Nodes

def generate_users(user_number, blocks, ground_x, ground_y):
    num_nodes = user_number
    user_nodes = []  # List to hold [x, y] coordinates of user nodes
    node_count = 0

    while node_count < user_number:
        x = random.random() * ground_x
        y = random.random() * ground_y
        node_status = 0

        for block in blocks:
            square_x_length = block['size'][0]
            square_y_length = block['size'][1]
            square_bottom_corner = block['bottomCorner']
            if (x >= square_bottom_corner[0] and x <= square_bottom_corner[0] + square_x_length) and \
                (y >= square_bottom_corner[1] and y <= square_bottom_corner[1] + square_y_length):
                node_status = 1
                break

        if node_status == 0:
            node_count += 1
            user_nodes.append([x, y])

    ground_users = []
    for node in user_nodes:
        # ground_users.append(Nodes([node[0], node[1], 0], 'ground users', 0))
        ground_users.append(Nodes((node[0], node[1], 0), 'ground users', 0))

    return ground_users