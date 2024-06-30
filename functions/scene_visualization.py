import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def scene_visualization(ground_users = None, UAV_nodes = None, air_base_station = None, blocks = None, scene_info = None, heatmap=None, min_height = 0, connection_GU_UAV=None, connection_UAV_BS=None, line_alpha=0, show_axes_labels=True):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # print(scene_info['xLength'])
    
    ax.set_xlim([0, scene_info['xLength']])
    ax.set_ylim([0, scene_info['yLength']])

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
    node_size = min(scene_info['xLength'], scene_info['yLength'])/40
    
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
    if connection_GU_UAV:
        for gu, path in connection_GU_UAV.items():
            start = gu
            end = path[0]+len(ground_users)
            start_pos = get_position_by_index(start, ground_users, UAV_nodes, air_base_station)
            end_pos = get_position_by_index(end, ground_users, UAV_nodes, air_base_station)

            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], color=line_color, alpha=line_alpha)
    
    if connection_UAV_BS:
        for uav, path in connection_UAV_BS.items():
            for i in range(len(path) - 1):
                start = path[i]+len(ground_users)
                end = path[i + 1]+len(ground_users)
                start_pos = get_position_by_index(start, ground_users, UAV_nodes, air_base_station)
                end_pos = get_position_by_index(end, ground_users, UAV_nodes, air_base_station)

                ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], color=line_color, alpha=line_alpha)

    plt.show()

def get_position_by_index(index, ground_users, UAV_nodes, air_base_station):
    all_nodes = (ground_users or []) + (UAV_nodes or []) + (air_base_station or [])
    if 0 <= index < len(all_nodes):
        node = all_nodes[index]
        return node.position
    return [0, 0, 0]

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

# def visualize_selected_heights_heatmaps(heatmap, heights, colormap='hot'):
#     """
#     可视化选定高度的2D热图。

#     参数：
#     heatmap (numpy.ndarray): 三维热图数据。
#     heights (list of int): 想要可视化的高度层索引列表。
#     colormap (str): 颜色映射，默认值为 'hot'。
#     """
#     num_plots = len(heights)
#     cols = int(np.ceil(np.sqrt(num_plots)))
#     rows = int(np.ceil(num_plots / cols))
    
#     fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
#     axes = axes.flatten()

#     for i, z in enumerate(heights):
#         if z < 0 or z >= heatmap.shape[2]:
#             print(f"Height {z} is out of bounds.")
#             axes[i].axis('off')
#             continue

#         im = axes[i].imshow(heatmap[:, :, z], cmap=colormap, interpolation='nearest')
#         axes[i].set_title(f'Height {z}')
#         axes[i].set_xlabel('X Axis')
#         axes[i].set_ylabel('Y Axis')
    
#     for ax in axes[num_plots:]:
#         ax.axis('off')

#     cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
#     cbar.set_label('Intensity')

#     plt.tight_layout()
#     plt.show()

def visualize_selected_heights_heatmaps(heatmap, heights, colormap='hot', min_height=0):
    """
    可视化选定高度的2D热图。

    参数：
    heatmap (numpy.ndarray): 三维热图数据。
    heights (list of int): 想要可视化的高度层索引列表。
    colormap (str): 颜色映射，默认值为 'hot'。
    min_height (int): 热图数据的最小高度，用于计算有效索引。
    """
    # 计算有效的高度索引
    valid_indices = [h - min_height for h in heights if min_height <= h < min_height + heatmap.shape[2]]
    valid_heights = [h for h in heights if min_height <= h < min_height + heatmap.shape[2]]
    
    if not valid_indices:
        print("No valid heights to display.")
        return
    
    num_plots = len(valid_indices)
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    
    # 如果只有一个子图，axes不是数组，需要转换为可迭代的列表
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (z_index, z_height) in enumerate(zip(valid_indices, valid_heights)):
        im = axes[i].imshow(heatmap[:, :, z_index], cmap=colormap, interpolation='nearest')
        axes[i].set_title(f'Height {z_height}')
        axes[i].set_xlabel('X Axis')
        axes[i].set_ylabel('Y Axis')
    
    # 隐藏多余的子图
    for ax in axes[num_plots:]:
        ax.axis('off')

    # 创建新的轴用于颜色条，并调整子图间距和边距
    fig.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.show()
# # 测试数据
# class User:
#     def __init__(self, position):
#         self.position = position

# ground_users = [User((10, 10)), User((20, 20))]
# UAV_nodes = [User((30, 30, 10)), User((40, 40, 20))]
# air_base_station = [User((50, 50, 30))]
# blocks = [{'bottomCorner': (0, 0, 0), 'size': (10, 10), 'height': 10}]
# scene_info = {'xLength': 100, 'yLength': 100}
# heatmap = np.random.randint(0, 5, (100, 100, 100))

# scene_visualization(ground_users, UAV_nodes, air_base_station, blocks, scene_info, heatmap)