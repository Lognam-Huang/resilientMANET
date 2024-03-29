import numpy as np

def path_is_blocked(blocks, nodeA, nodeB):
    for cube in blocks:  # 遍历所有的 blocks
        bottom_corner = cube['bottomCorner']
        width, length = cube['size']  # 假设 size 包含了宽度、长度
        height = cube['height']

        # print(type(bottom_corner))
        # suppose all block is on the ground, that is, z = 0
        # bottom_corner[2] = 0
        # bottom_corner.append(0)
        # print(bottom_corner)

        # define 6 faces of the cube
        faces = [
            ((bottom_corner[0], bottom_corner[1], bottom_corner[2]), (0, 0, 1)),  # bottom
            ((bottom_corner[0], bottom_corner[1], bottom_corner[2] + height), (0, 0, -1)),  # top
            ((bottom_corner[0], bottom_corner[1], bottom_corner[2]), (0, 1, 0)),  # front
            ((bottom_corner[0], bottom_corner[1] + length, bottom_corner[2]), (0, -1, 0)),  # back
            ((bottom_corner[0], bottom_corner[1], bottom_corner[2]), (1, 0, 0)),  # left
            ((bottom_corner[0] + width, bottom_corner[1], bottom_corner[2]), (-1, 0, 0)),  # right
            
            # ((bottom_corner[0], bottom_corner[1], 0), (0, 0, 1)),  # bottom
            # ((bottom_corner[0], bottom_corner[1], 0 + height), (0, 0, -1)),  # top
            # ((bottom_corner[0], bottom_corner[1], 0), (0, 1, 0)),  # front
            # ((bottom_corner[0], bottom_corner[1] + length, 0), (0, -1, 0)),  # back
            # ((bottom_corner[0], bottom_corner[1], 0), (1, 0, 0)),  # left
            # ((bottom_corner[0] + width, bottom_corner[1], 0), (-1, 0, 0)),  # right
        ]

        for plane_point, plane_normal in faces:
            intersection = intersect_plane(nodeA.position, nodeB.position, plane_point, plane_normal)
            if intersection and is_point_inside_cube(intersection, (bottom_corner[0], bottom_corner[1], bottom_corner[2]), (bottom_corner[0] + width, bottom_corner[1] + length, bottom_corner[2] + height)):
                return True  # 如果找到任何阻塞，则返回 True

    return False  # 如果没有找到阻塞，返回 False

def intersect_plane(p0, p1, plane_point, plane_normal):
    line_vector = tuple(p1[i] - p0[i] for i in range(3))
    dot_product = sum(plane_normal[i] * line_vector[i] for i in range(3))
    
    if abs(dot_product) < 1e-6:
        return None  # 线段与平面平行，不相交
    
    t = sum(plane_normal[i] * (plane_point[i] - p0[i]) for i in range(3)) / dot_product
    if 0 <= t <= 1:
        return tuple(p0[i] + t * line_vector[i] for i in range(3))
    
    return None

def is_point_inside_cube(p, cube_min, cube_max):
    return all(cube_min[i] <= p[i] <= cube_max[i] for i in range(3))


# import numpy as np

# def path_is_blocked(blocks, nodeA, nodeB):
#     # for block in blocks:
#     #     print(block)
    
#     # print(nodeA)
#     # print(nodeB)
    
#     cube = blocks[0]
    
#     bottom_corner = cube['bottomCorner']
#     width, length = cube['size']
#     height = cube['height']

#     # define 6 faces of the rectangle
#     faces = [
#         ((bottom_corner[0], bottom_corner[1], 0), (0, 0, 1)),  # bottom
#         ((bottom_corner[0], bottom_corner[1], height), (0, 0, -1)),  # top
#         ((bottom_corner[0], bottom_corner[1], 0), (0, 1, 0)),  # front
#         ((bottom_corner[0], bottom_corner[1] + length, 0), (0, -1, 0)),  # back
#         ((bottom_corner[0], bottom_corner[1], 0), (1, 0, 0)),  # left
#         ((bottom_corner[0] + width, bottom_corner[1], 0), (-1, 0, 0)),  # right
#     ]

#     for plane_point, plane_normal in faces:
#         intersection = intersect_plane(nodeA.position, nodeB.position, plane_point, plane_normal)
#         if intersection and is_point_inside_cube(intersection, (bottom_corner[0], bottom_corner[1], 0), (bottom_corner[0] + width, bottom_corner[1] + length, height)):
#             return True

#     return False

# def intersect_plane(p0, p1, plane_point, plane_normal):
#     # check if the line segment intersects the plane, and return the point of intersection
#     line_vector = tuple(p1[i] - p0[i] for i in range(3))
#     dot_product = sum(plane_normal[i] * line_vector[i] for i in range(3))
    
#     if abs(dot_product) < 1e-6:
#         # the line segments are parallel to the plane and do not intersect
#         return None
    
#     t = sum(plane_normal[i] * (plane_point[i] - p0[i]) for i in range(3)) / dot_product
#     if 0 <= t <= 1:
#         # the intersection point is on the line segment
#         return tuple(p0[i] + t * line_vector[i] for i in range(3))
#     return None


# def is_point_inside_cube(p, cube_min, cube_max):
#     # check if the point is inside the rectangle
#     return all(cube_min[i] <= p[i] <= cube_max[i] for i in range(3))