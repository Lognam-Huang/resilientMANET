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

    node_number = 0

    for node in user_nodes:
        # ground_users.append(Nodes([node[0], node[1], 0], 'ground users', 0))
        ground_users.append(Nodes((node[0], node[1], 0), 'ground users', 0, node_number))
        node_number += 1

    return ground_users

def add_new_users(existing_users, total_user_number=0, extra_user_number=0, blocks=None, ground_x=None, ground_y=None):
    new_users = []
    node_count = 0
    max_node_number = max(user.node_number for user in existing_users) + 1

    if total_user_number >0 and extra_user_number>0:
        print("Conflict GU increasement condition")
    elif total_user_number > 0:
        while node_count < total_user_number:
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
                new_users.append([x, y])

        for node in new_users:
            existing_users.append(Nodes((node[0], node[1], 0), 'ground users', 0, max_node_number))
            max_node_number += 1
    # elif extra_user_number>0:
    #     for i in range(extra_user_number):

    return existing_users

def add_or_remove_GU(ground_users, blocks, x_length, y_length, max_movement_distance, n_change, add=True, print_info=False):    
    # if print_info:
    #     print("Ground users after movement:")
    #     for gu in ground_users:
    #         print(f"GU {gu.node_number}: ({gu.position[0]}, {gu.position[1]})")
    
    if add:
        # 增加n个新的GU
        node_count = 0
        max_node_number = max(user.node_number for user in ground_users) + 1
        new_users = []
        
        while node_count < n_change:
            x = random.random() * x_length
            y = random.random() * y_length
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
                new_users.append([x, y])
        
        for node in new_users:
            ground_users.append(Nodes((node[0], node[1], 0), 'ground users', 0, max_node_number))
            max_node_number += 1
            
        if print_info:
            print(f"{n_change} new ground users added:")
            for gu in new_users:
                print(f"New GU: ({gu[0]}, {gu[1]})")
    else:
        # 减少n个GU
        if n_change > len(ground_users):
            raise ValueError("The number of ground users to remove exceeds the total number of ground users.")
        
        removed_users = random.sample(ground_users, n_change)
        for user in removed_users:
            ground_users.remove(user)
        
        if print_info:
            print(f"{n_change} ground users removed.")
            for gu in removed_users:
                print(f"Removed GU {gu.node_number}: ({gu.position[0]}, {gu.position[1]})")

    # return ground_users
