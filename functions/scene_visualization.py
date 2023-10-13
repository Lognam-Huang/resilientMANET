import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def scene_visualization(ground_users, UAV_nodes, air_base_station, blocks, scene_info):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # print(scene_info['xLength'])
    
    ax.set_xlim([0, scene_info['xLength']])
    ax.set_ylim([0, scene_info['yLength']])
    # ax.set_zlim([0, 600])
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # set box size of each UAV_nodes
    node_size = 30
    
    # for block in blocks:
    #     x, y = block['bottomCorner']
    #     dx, dy = block['size']
    #     dz = block['height']
    #     color = block['color']
    #     label = block['label']
        
    #     ax.bar3d(x, y, 0, dx, dy, dz, shade=True, color=color)
        
    #     # 添加标记文字在块的中心
    #     ax.text(x + dx/2, y + dy/2, dz/2, label, color='black', ha='center', va='center')
    
    for block in blocks:                
        x, y = block['bottomCorner']
        dx, dy = block['size']
        dz = block['height']
        color = (1, 1, 1, 0.5)
        # label = block['label']
        
        ax.bar3d(x, y, 0, dx, dy, dz, shade=True, color=color)
        
    for user in ground_users:        
        # print(user.position[0])
        
        x, y = user.position[0], user.position[1]
        dx, dy, dz= node_size, node_size, node_size
        color = (0, 0, 1, 0.5)
        # label = block['label']
        
        ax.bar3d(x, y, 0, dx, dy, dz, shade=True, color=color)
        
    for UAV in UAV_nodes:        
        x, y, z = UAV.position[0], UAV.position[1], UAV.position[2]
        dx, dy, dz= node_size, node_size, node_size
        color = (0, 1, 0, 0.5)
        # label = block['label']
        
        ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=color)


    for ABS in air_base_station:        
        x, y, z = ABS.position[0], ABS.position[1], ABS.position[2]
        dx, dy, dz= node_size, node_size, node_size
        color = (1, 0, 0, 0.5)
        # label = block['label']
        
        ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=color)


    plt.show()