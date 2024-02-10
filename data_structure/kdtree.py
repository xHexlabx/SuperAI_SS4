import numpy as np

class Node:
    
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

class KDTree:
    
    def __init__(self, points):
        self.root = self.build_tree(points)

    def build_tree(self, points, depth=0):
        if not points:
            return None

        k = len(points[0])
        axis = depth % k

        points.sort(key=lambda x: x[axis])
        median = len(points) // 2

        return Node(
            points[median],
            self.build_tree(points[:median], depth + 1),
            self.build_tree(points[median + 1:], depth + 1)
        )

    def nearest_neighbor(self, target):
        best = None
        best_dist = float('inf')

        def recursive_search(node, depth=0):
            nonlocal best, best_dist

            if node is None:
                return

            k = len(target)
            axis = depth % k

            branch = None
            opposite_branch = None

            if target[axis] < node.point[axis]:
                branch = node.left
                opposite_branch = node.right
            else:
                branch = node.right
                opposite_branch = node.left

            recursive_search(branch, depth + 1)

            dist = np.linalg.norm(np.array(target) - np.array(node.point))
            if dist < best_dist:
                best = node.point
                best_dist = dist

            if abs(target[axis] - node.point[axis]) < best_dist:
                recursive_search(opposite_branch, depth + 1)

        recursive_search(self.root)
        return best

points = [(3 , 6) , (2 , 7) ,(17,15) , (6 , 12) , (13 , 15) , (9 , 1) , (10 , 19)]
tree = KDTree(points)
print(tree.nearest_neighbor((10 , 10)))  
