import numpy as np

from functions.path_is_blocked import path_is_blocked
from classes.Nodes import Nodes

def get_3D_heatmap(ground_users, blocks, scene_info, min_height, max_height):
    x_length = scene_info['xLength']
    y_length = scene_info['yLength']
    # 初始化一个三维数组来存储每个点的得分
    heatmap = np.zeros((x_length, y_length, max_height - min_height + 1))
    
    # 遍历每个高度
    for z in range(min_height, max_height + 1):
        # 遍历每个x和y坐标
        for x in range(x_length):
            for y in range(y_length):
                for user in ground_users:
                    # 检查从当前位置到ground_user的直视线是否被阻挡
                    # print(user)
                    curPos = Nodes([x,y,z])
                    # curPos.position = (x,y,z)
                    if not path_is_blocked(blocks,  curPos,  user):
                        # 如果未被阻挡，增加该点的得分
                        heatmap[x, y, z - min_height] += 1
    return heatmap



