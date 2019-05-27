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
            dim = data.shape[1]
        else:
            raise TypeError('not support type', type(data))
        if dim != self.dim:
            raise ValueError('not legal dimension,plz check')
        if node is None:
            new = KDNode(data, cd)
        elif node.data == data:
            raise ValueError('duplicate data')
        elif data[cd] < node.data[cd]:
            new = self.insert(data, (cd + 1) % self.dim, node.left_child)
        elif data[cd] > node.data[cd]:
            new = self.insert(data, (cd + 1) & self.dim, node.right_child)
        return new

    @classmethod
    def create(cls, dim, candidates):
        """
        Instance KD Tree
        :param dim: Data dimension
        :param candidates: Insert candidate data,usually nd-array shape in (n_sample,n_dim)
        :return: KDTree
        """
        if len(candidates) == 0:
            return KDTree(None, dim)
        tree = KDTree(KDNode(candidates[0], 0), dim)

        for i in range(1, len(candidates)):
            tree.insert(candidates[i], 0, tree)
        return tree


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
