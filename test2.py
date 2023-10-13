# import ipyvolume as ipv

# # 你的数据
# blocks = [
#     {'bottomCorner': [350, 380], 'size': [80, 80], 'height': 100},
#     {'bottomCorner': [350, 10], 'size': [60, 80], 'height': 200},
#     {'bottomCorner': [20, 570], 'size': [100, 80], 'height': 400}
# ]
# scene = {'xLength': 500, 'yLength': 700}
# groundUserPosition = (136.0598323772384, 398.166713839298, 0)

# # 绘制blocks
# for block in blocks:
#     x, y = block['bottomCorner']
#     dx, dy = block['size']
#     dz = block['height']
#     ipv.plot_mesh([x, x+dx, x+dx, x, x, x+dx, x+dx, x],
#                 [y, y, y+dy, y+dy, y, y, y+dy, y+dy], 
#                 [0, 0, 0, 0, dz, dz, dz, dz], 
#                 triangles=[[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 3, 7], [0, 4, 7], [1, 2, 6], [1, 5, 6], [0, 1, 5], [0, 4, 5], [2, 3, 7], [2, 6, 7]], 
#                 color="blue")

# # 绘制groundUser
# ipv.scatter(*groundUserPosition, color="red", marker="sphere", size=5)

# # 显示
# ipv.show()

# import ipyvolume as ipv

# # 你的数据
# blocks = [
#     {'bottomCorner': [350, 380], 'size': [80, 80], 'height': 100},
#     {'bottomCorner': [350, 10], 'size': [60, 80], 'height': 200},
#     {'bottomCorner': [20, 570], 'size': [100, 80], 'height': 400}
# ]
# scene = {'xLength': 500, 'yLength': 700}
# groundUserPosition = (136.0598323772384, 398.166713839298, 0)

# # 绘制blocks
# for block in blocks:
#     x, y = block['bottomCorner']
#     dx, dy = block['size']
#     dz = block['height']
#     ipv.plot_trisurf([x, x+dx, x+dx, x, x, x+dx, x+dx, x], 
#                     [y, y, y+dy, y+dy, y, y, y+dy, y+dy], 
#                     [0, 0, 0, 0, dz, dz, dz, dz], 
#                     triangles=[[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 3, 7], [0, 4, 7], [1, 2, 6], [1, 5, 6], [0, 1, 5], [0, 4, 5], [2, 3, 7], [2, 6, 7]], 
#                     color="blue")

# # 绘制groundUser
# ipv.scatter(*groundUserPosition, color="red", marker="sphere", size=5)

# # 显示
# ipv.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 你的数据，增加了颜色、透明度和标记
blocks = [
    {'bottomCorner': [350, 380], 'size': [80, 80], 'height': 100, 'color': (1, 0, 0, 0.5), 'label': 'Block 1'},
    {'bottomCorner': [350, 10], 'size': [60, 80], 'height': 200, 'color': (0, 1, 0, 0.7), 'label': 'Block 2'},
    {'bottomCorner': [20, 570], 'size': [100, 80], 'height': 400, 'color': (0, 0, 1, 0.9), 'label': 'Block 3'}
]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([0, 600])
ax.set_ylim([0, 600])
ax.set_zlim([0, 600])

for block in blocks:
    x, y = block['bottomCorner']
    dx, dy = block['size']
    dz = block['height']
    color = block['color']
    label = block['label']
    
    ax.bar3d(x, y, 0, dx, dy, dz, shade=True, color=color)
    
    # 添加标记文字在块的中心
    ax.text(x + dx/2, y + dy/2, dz/2, label, color='black', ha='center', va='center')

plt.show()
