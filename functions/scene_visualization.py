import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def scene_visualization(ground_users, UAV_nodes = None, air_base_station = None, blocks = None, scene_info = None, heatmap=None, min_height = 0):

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
    node_size = 2
    
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
            
            ax.bar3d(x, y, 0, dx, dy, dz, shade=True, color=color)

    if UAV_nodes:                 
        for UAV in UAV_nodes:        
            x, y, z = UAV.position[0], UAV.position[1], UAV.position[2]
            dx, dy, dz= node_size, node_size, node_size
            color = (0, 1, 0, 0.5)
            # label = block['label']
            
            ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=color)


    if air_base_station:
        for ABS in air_base_station:        
            x, y, z = ABS.position[0], ABS.position[1], ABS.position[2]
            dx, dy, dz= node_size, node_size, node_size
            color = (1, 0, 0, 0.5)
            # label = block['label']
            
            ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=color)
    
    # 可视化heatmap
    if heatmap is not None:
        max_users = np.max(heatmap)  # 获取最大的ground_user数
        for x in range(heatmap.shape[0]):
            for y in range(heatmap.shape[1]):
                for z in range(heatmap.shape[2]):
                    value = heatmap[x, y, z]
                    if value > 0:  # 如果该点的值大于0
                        alpha = (value / max_users) * 0.02  # 根据ground_user的数量调整透明度
                        ax.bar3d(x, y, z+min_height, 1, 1, 1, shade=False, color=(0, 1, 0, alpha))



    plt.show()


# def visualize_2D_heatmap_per_layer(heatmap, min_height, max_height):
#     layers = heatmap.shape[2]  # Z轴的层数
#     cols = int(np.ceil(np.sqrt(layers)))  # 取根号下的层数向上取整作为列数
#     rows = layers // cols + (1 if layers % cols else 0)  # 确保有足够的行数来展示所有层

#     fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))  # 调整子图大小
#     axes = np.array(axes).reshape(-1)  # 确保axes总是一个数组，方便单层时处理

#     for z in range(layers):
#         if rows * cols == 1:
#             ax = axes
#         else:
#             ax = axes[z]
#         cax = ax.imshow(heatmap[:, :, z], cmap='viridis', interpolation='nearest')
#         ax.set_title(f'Height {z + min_height}')
#         ax.set_xlabel('X Axis')
#         ax.set_ylabel('Y Axis')
    
#     # 隐藏不使用的子图
#     for z in range(layers, len(axes)):
#         axes[z].axis('off')

#     # 在子图下方添加颜色条
#     cbar = fig.colorbar(cax, ax=axes, orientation='horizontal', pad=0.1, fraction=0.02)
#     cbar.set_label('Intensity')

#     plt.tight_layout()
#     plt.subplots_adjust(bottom=0.2)  # 调整子图位置，为颜色条留出空间
#     plt.show()
    
def visualize_2D_heatmap_combined(heatmap, min_height, max_height = 0, colormap='hot'):
    layers = heatmap.shape[2]  # Z轴的层数
    cols = int(np.ceil(np.sqrt(layers)))  # 计算列数
    rows = layers // cols + (1 if layers % cols else 0)  # 计算行数

    # 增加图形尺寸
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))  # 调整figsize以提供更多空间
    if rows * cols > 1:
        axes = axes.ravel()  # 扁平化处理，便于迭代
    else:
        axes = [axes]  # 确保axes是可迭代的

    for z, ax in zip(range(layers), axes):
        cax = ax.imshow(heatmap[:, :, z], cmap=colormap, interpolation='nearest')
        ax.set_title(f'Height {z + min_height}')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
    
    # 隐藏不使用的子图
    for ax in axes[z+1:]:
        ax.axis('off')

    # 在子图下方添加颜色条
    cbar = fig.colorbar(cax, ax=axes, orientation='horizontal', pad=0.1, fraction=0.02)
    cbar.set_label('Intensity')

    # fig.colorbar(cax, ax=axes, orientation='horizontal', pad=0.1, fraction=0.02, aspect=40, shrink=0.8)
    # cbar.set_label('Intensity')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.4, hspace=0.4)  # 手动调整子图间距
    plt.show()
