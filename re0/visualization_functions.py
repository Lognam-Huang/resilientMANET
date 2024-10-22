import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def scene_visualization(ground_users = None, UAV_nodes = None, air_base_station = None, scene_info = None,line_alpha=0, show_axes_labels=True):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # print(scene_info['xLength'])
    blocks = scene_info['blocks']
    # print(scene_info['scenario'])
    
    ax.set_xlim([0, scene_info['scenario']['xLength']])
    ax.set_ylim([0, scene_info['scenario']['yLength']])

    max_block_height = 0
    
    max_block_height = max((block['height'] for block in blocks), default=0) if blocks else 0
    max_uav_height = max((UAV.position[2] for UAV in UAV_nodes), default=0) if UAV_nodes else 0
    max_abs_height = max((ABS.position[2] for ABS in air_base_station), default=0) if air_base_station else 0
    
    max_height = max(max_block_height, max_uav_height, max_abs_height) * 1.2  # 取最大值并乘以120%
    ax.set_zlim([0, max_height])
    
    
    if show_axes_labels:
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
    else:
        # ax.set_xticks(ax.get_xticks())
        # ax.set_yticks(ax.get_yticks())
        # ax.set_zticks(ax.get_zticks())
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        # ax.set_xlabel('')
        # ax.set_ylabel('')
        # ax.set_zlabel('')

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # set box size of each UAV_nodes
    # node_size = 2
    node_size = min(scene_info['scenario']['xLength'], scene_info['scenario']['yLength'])/40
    
    if blocks:    
        for block in blocks:    
            # print(block)
            # print(block['bottomCorner'])            
            x, y, z = block['bottomCorner']
            dx, dy = block['size']
            dz = block['height']
            color = (1, 1, 1, 0.5)
            # label = block['label']
            
            ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=color)

            # 添加标记文字在块的中心
            # ax.text(x + dx/2, y + dy/2, dz/2, label, color='black', ha='center', va='center') 
        
    if ground_users:            
        for user in ground_users:        
            # print(user.position[0])
        
            x, y = user.position[0], user.position[1]
            dx, dy, dz= node_size, node_size, node_size
            color = (0, 0, 1, 0.5)
            # label = block['label']

            # print("GU")
            # print(x, y, 0, dx, dy, dz)
            
            ax.bar3d(x, y, 0, dx, dy, dz, shade=True, color=color)

    if UAV_nodes:                 
        for UAV in UAV_nodes:        
            x, y, z = UAV.position[0], UAV.position[1], UAV.position[2]
            dx, dy, dz= node_size, node_size, node_size
            color = (0, 1, 0, 0.5)
            # label = block['label']

            # print("UAV")
            # print(x, y, z, dx, dy, dz)
            
            ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=color)


    if air_base_station:
        for ABS in air_base_station:        
            x, y, z = ABS.position[0], ABS.position[1], ABS.position[2]
            dx, dy, dz= node_size, node_size, node_size
            color = (1, 0, 0, 0.5)
            # label = block['label']
            
            ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=color)
    
    # # 可视化heatmap
    # if heatmap is not None:
    #     max_users = np.max(heatmap)  # 获取最大的ground_user数
    #     for x in range(heatmap.shape[0]):
    #         for y in range(heatmap.shape[1]):
    #             for z in range(heatmap.shape[2]):
    #                 value = heatmap[x, y, z]
    #                 if value > 0:  # 如果该点的值大于0
    #                     alpha = (value / max_users) * 0.02  # 根据ground_user的数量调整透明度
    #                     ax.bar3d(x, y, z+min_height, 1, 1, 1, shade=False, color=(0, 1, 0, alpha))

    # if connection_GU_UAV and ground_users and UAV_nodes:
    #     for gu_index, uav_index in connection_GU_UAV:
    #         gu = ground_users[gu_index]
    #         uav = UAV_nodes[uav_index]
    #         gu_pos = np.array([gu.position[0], gu.position[1], 0])
    #         uav_pos = np.array([uav.position[0], uav.position[1], uav.position[2]])
    #         ax.plot([gu_pos[0], uav_pos[0]], [gu_pos[1], uav_pos[1]], [gu_pos[2], uav_pos[2]], color='k')

    # if connection_GU_UAV:
    #     for start, end in connection_GU_UAV:

    #         print(start)
    #         print(end)

    #         start_pos = get_position_by_index(start, ground_users, UAV_nodes, air_base_station)
    #         end_pos = get_position_by_index(end, ground_users, UAV_nodes, air_base_station)
    #         ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], color='k')


    line_color = (0.5,0,0)
    
    from node_functions import print_node
    # print_node(UAV_nodes)
    
    for gu in ground_users:
        gu_x, gu_y, gu_z = gu.position[0], gu.position[1], gu.position[2]
        uav_index =  gu.connected_nodes[0]
        uav_x, uav_y, uav_z = UAV_nodes[uav_index].position[0], UAV_nodes[uav_index].position[1], UAV_nodes[uav_index].position[2]

        ax.plot([gu_x, uav_x], [gu_y, uav_y], [gu_z, uav_z], color=(0.5,0,0), alpha=line_alpha)
    
    for uav in UAV_nodes:
        start_uav_x, start_uav_y, start_uav_z = uav.position[0], uav.position[1], uav.position[2]
        for target_uav_index in uav.connected_nodes:
            # print(target_uav_index)
            target_uav_x, target_uav_y, target_uav_z = UAV_nodes[target_uav_index].position[0], UAV_nodes[target_uav_index].position[1], UAV_nodes[target_uav_index].position[2]

            ax.plot([start_uav_x, target_uav_x], [start_uav_y, target_uav_y], [start_uav_z, target_uav_z], color=(0.5,0,0), alpha=line_alpha)
    
    for bs in air_base_station:
        bs_x, bs_y, bs_z = bs.position[0], bs.position[1], bs.position[2]
        for target_uav_index in bs.connected_nodes:
            target_uav_x, target_uav_y, target_uav_z = UAV_nodes[target_uav_index].position[0], UAV_nodes[target_uav_index].position[1], UAV_nodes[target_uav_index].position[2]

            ax.plot([bs_x, target_uav_x], [bs_y, target_uav_y], [bs_z, target_uav_z], color=(0.5,0,0), alpha=line_alpha)
        
    # GU-UAV连接线
    # if ground_users and UAV_nodes:
    #     for user in ground_users:
    #         # print(user.position)
    #         gu_x, gu_y, gu_z = user.position[0], user.position[1], 0  # Ground users 的 z 坐标为 0
    #         for UAV in UAV_nodes:  # 假设所有 GU 都与 UAV 相连
    #             print(UAV.position)
    #             uav_x, uav_y, uav_z = UAV.position[0], UAV.position[1], UAV.position[2]
    #             ax.plot([gu_x, uav_x], [gu_y, uav_y], [gu_z, uav_z], color=line_color, alpha=line_alpha)

    # UAV-UAV连接线
    # if UAV_nodes:
    #     for i, UAV1 in enumerate(UAV_nodes):
    #         for j, UAV2 in enumerate(UAV_nodes):
    #             if i < j:  # 避免重复连接
    #                 uav1_x, uav1_y, uav1_z = UAV1.position[0], UAV1.position[1], UAV1.position[2]
    #                 uav2_x, uav2_y, uav2_z = UAV2.position[0], UAV2.position[1], UAV2.position[2]
    #                 ax.plot([uav1_x, uav2_x], [uav1_y, uav2_y], [uav1_z, uav2_z], color=line_color, alpha=line_alpha)

    # # UAV-BS连接线
    # if UAV_nodes and air_base_station:
    #     for UAV in UAV_nodes:
    #         uav_x, uav_y, uav_z = UAV.position[0], UAV.position[1], UAV.position[2]
    #         for ABS in air_base_station:
    #             abs_x, abs_y, abs_z = ABS.position[0], ABS.position[1], ABS.position[2]
    #             ax.plot([uav_x, abs_x], [uav_y, abs_y], [uav_z, abs_z], color=line_color, alpha=line_alpha)

    # if connection_GU_UAV:
    #     for gu, path in connection_GU_UAV.items():
    #         if path:
    #             start = gu
    #             end = path[0]+len(ground_users)
    #             start_pos = get_position_by_index(start, ground_users, UAV_nodes, air_base_station)
    #             end_pos = get_position_by_index(end, ground_users, UAV_nodes, air_base_station)

    #             ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], color=line_color, alpha=line_alpha)
    
    # if connection_UAV_BS:
    #     for uav, path in connection_UAV_BS.items():
    #         for i in range(len(path) - 1):
    #             start = path[i]+len(ground_users)
    #             end = path[i + 1]+len(ground_users)
    #             start_pos = get_position_by_index(start, ground_users, UAV_nodes, air_base_station)
    #             end_pos = get_position_by_index(end, ground_users, UAV_nodes, air_base_station)

    #             ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], color=line_color, alpha=line_alpha)

    plt.show()

def get_position_by_index(index, ground_users, UAV_nodes, air_base_station):
    all_nodes = (ground_users or []) + (UAV_nodes or []) + (air_base_station or [])
    if 0 <= index < len(all_nodes):
        node = all_nodes[index]
        return node.position
    return [0, 0, 0]