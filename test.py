# def is_blocked_by_box(A, B, box):
#     # Step 1: 判断两点是否在长方体的同一侧
#     for dim in range(3):
#         if (A[dim] < box['bottomCorner'][dim] and B[dim] < box['bottomCorner'][dim]) or \
#             (A[dim] > box['bottomCorner'][dim] + (box['size'][dim] if dim < 2 else box['height']) and
#             B[dim] > box['bottomCorner'][dim] + (box['size'][dim] if dim < 2 else box['height'])):
#             return False

#     # Step 2: 判断两点之间的线段是否与长方体的任意面相交
#     # 这里只是一个简化的判断，为了真正的判断需要更复杂的计算
#     for dim in range(2):
#         if (A[dim] <= box['bottomCorner'][dim] <= B[dim] or B[dim] <= box['bottomCorner'][dim] <= A[dim]) and \
#             (A[2] <= box['bottomCorner'][2] <= B[2] or B[2] <= box['bottomCorner'][2] <= A[2]):
#             return True
#         if (A[dim] <= box['bottomCorner'][dim] + box['size'][dim] <= B[dim] or
#             B[dim] <= box['bottomCorner'][dim] + box['size'][dim] <= A[dim]) and \
#             (A[2] <= box['bottomCorner'][2] + box['height'] <= B[2] or
#             B[2] <= box['bottomCorner'][2] + box['height'] <= A[2]):
#             return True
#     return False

# def is_blocked(A, B, blocks):
#     for box in blocks:
#         if is_blocked_by_box(A, B, box):
#             return True
#     return False

# # 测试数据
# blocks = [
#     {'bottomCorner': [350, 380, 0], 'size': [80, 80], 'height': 100},
#     {'bottomCorner': [350, 10, 0], 'size': [60, 80], 'height': 200},
#     {'bottomCorner': [20, 570, 0], 'size': [100, 80], 'height': 400}
# ]

# A = [0, 0, 0]
# B = [200,200,0]

# print(is_blocked(A, B, blocks))  # 返回值表示A和B之间是否被长方体阻挡


class UAVMap:
    def __init__(self, AllPaths):
        self.AllPaths = AllPaths
    
    def quantify_data_rate(self, r):
        print(self.AllPaths.values())
        # 1. 获取每个元素的最大DR
        max_data_rates = [max(paths, key=lambda x: x['DR'])['DR'] if paths else 0 for paths in self.AllPaths.values()]
        
        print(max_data_rates)
        
        # 2. 计算所有元素的最小DR和平均DR
        min_DR = min(max_data_rates)
        avg_DR = sum(max_data_rates) / len(max_data_rates)
        
        # 3. 使用公式计算score
        score = r * min_DR + (1 - r) * avg_DR
        return score

# 示例
uav_map = UAVMap(AllPaths={0: [{'path': [0, 2, 6], 'DR': 1331604646.8945067}, {'path': [0, 6], 'DR': 868893401.0471452}], 1: [{'path': [1, 0, 2, 6], 'DR': 312893737.12986887}, {'path': [1, 0, 6], 'DR': 312893737.12986887}], 2: [{'path': [2, 0, 6], 'DR': 868893401.0471452}, {'path': [2, 6], 'DR': 2008770280.465467}], 3: [{'path': [3, 5], 'DR': 1745297098.210253}], 4: [{'path': [4, 6], 'DR': 2273496381.465522}]
                        # 5: []
                        })

r = 0.5  # 可以根据需要改变
score = uav_map.quantify_data_rate(r)
print(score)
