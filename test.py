def is_blocked_by_box(A, B, box):
    # Step 1: 判断两点是否在长方体的同一侧
    for dim in range(3):
        if (A[dim] < box['bottomCorner'][dim] and B[dim] < box['bottomCorner'][dim]) or \
            (A[dim] > box['bottomCorner'][dim] + (box['size'][dim] if dim < 2 else box['height']) and
            B[dim] > box['bottomCorner'][dim] + (box['size'][dim] if dim < 2 else box['height'])):
            return False

    # Step 2: 判断两点之间的线段是否与长方体的任意面相交
    # 这里只是一个简化的判断，为了真正的判断需要更复杂的计算
    for dim in range(2):
        if (A[dim] <= box['bottomCorner'][dim] <= B[dim] or B[dim] <= box['bottomCorner'][dim] <= A[dim]) and \
            (A[2] <= box['bottomCorner'][2] <= B[2] or B[2] <= box['bottomCorner'][2] <= A[2]):
            return True
        if (A[dim] <= box['bottomCorner'][dim] + box['size'][dim] <= B[dim] or
            B[dim] <= box['bottomCorner'][dim] + box['size'][dim] <= A[dim]) and \
            (A[2] <= box['bottomCorner'][2] + box['height'] <= B[2] or
            B[2] <= box['bottomCorner'][2] + box['height'] <= A[2]):
            return True
    return False

def is_blocked(A, B, blocks):
    for box in blocks:
        if is_blocked_by_box(A, B, box):
            return True
    return False

# 测试数据
blocks = [
    {'bottomCorner': [350, 380, 0], 'size': [80, 80], 'height': 100},
    {'bottomCorner': [350, 10, 0], 'size': [60, 80], 'height': 200},
    {'bottomCorner': [20, 570, 0], 'size': [100, 80], 'height': 400}
]

A = [0, 0, 0]
B = [200,200,0]

print(is_blocked(A, B, blocks))  # 返回值表示A和B之间是否被长方体阻挡
