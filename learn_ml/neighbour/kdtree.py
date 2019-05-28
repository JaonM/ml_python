"""
K-D Tree Implementation
"""
import numpy as np


class KDTree(object):
    """
    Similar to binary search tree

    A data structure to split the data space

    Attributes:
        root: root node of the tree
        dim: data dimension
    """

    def __init__(self, root=None, dim=0):
        self.root = root
        self.dim = dim

    def insert(self, data, cd, node):
        if isinstance(data, (list, set, tuple)):
            dim = len(data)
        elif isinstance(data, np.ndarray):
            dim = data.shape[0]
        else:
            raise TypeError('not support type', type(data))
        if dim != self.dim:
            raise ValueError('not legal dimension,plz check')
        if node is None:
            node = KDNode(data, cd)
        elif (node.data == data).all():
            raise ValueError('duplicate data')
        elif data[cd] <= node.data[cd]:
            node.left_child = self.insert(data, (cd + 1) % self.dim, node.left_child)
        elif data[cd] > node.data[cd]:
            node.right_child = self.insert(data, (cd + 1) % self.dim, node.right_child)
        return node

    @classmethod
    def create(cls, dim, candidates):
        """
        Instance KD Tree
        :param dim: Data dimension
        :param candidates: Insert candidate data,usually nd-array shape in (n_sample,n_dim)
        :return: KDTree
        """
        if candidates is None or len(candidates) == 0:
            return KDTree(None, dim)
        if not isinstance(candidates, np.ndarray):
            candidates = np.array(candidates)
        tree = KDTree(KDNode(candidates[0], 0), dim)

        for i in range(1, len(candidates)):
            tree.insert(candidates[i], 0, tree.root)
        return tree

    def __str__(self):
        arr = list()
        self._print_traverse(self.root, arr)
        res = ''
        for node in arr:
            res += str(node)+' '
        return res[:-1]

    def _print_traverse(self, node, arr):
        if node is not None:
            arr.append(node)
        else:
            return
        if node.left_child is not None:
            self._print_traverse(node.left_child, arr)
        if node.right_child is not None:
            self._print_traverse(node.right_child, arr)
        return


class KDNode(object):
    """
    KDTree node

    Attributes:
        data: data entity,can be any dimension
        cd: cutting dimension
        left_child: left child node
        right_child: right child node
    """

    def __init__(self, data, cd, left_child=None, right_child=None):
        self.data = data
        self.cd = cd
        self.left_child = left_child
        self.right_child = right_child

    def __str__(self):
        if isinstance(self.data, int):
            return "Node{data:" + str(self.data) + ",cd:" + str(self.cd) + '}'
        else:
            return "Node{data shape:" + str(self.data.shape) + ",cd:" + str(self.cd) + '}'


if __name__ == "__main__":
    candidates = np.random.random((7, 10))
    print(candidates)
    kdtree = KDTree.create(candidates.shape[1], None)
    print(kdtree)
    print(kdtree.dim)
