"""
KD-Tree Test
"""
from learn_ml.neighbour.kdtree import KDTree
import numpy as np


def test_create_kdtree():
    candidates = np.random.rand(4, 10)
    kdtree = KDTree.create(10, candidates,labels=np.zeros(4))
    assert kdtree
    assert (kdtree.root.data == candidates[0]).all()


def test_find_min():
    candidates = np.random.rand(3, 3)
    kdtree = KDTree.create(3, candidates,labels=np.zeros(3))
    min_node = kdtree.find_min(0)
    assert min_node.data[0] == min(candidates[:, 0])


def test_delete_node():
    candidates = np.random.rand(3, 10)
    kdtree = KDTree.create(10, candidates,labels=np.zeros(3))
    origin_len = kdtree.get_node_num()
    kdtree.delete_node(candidates[2])
    new_len = kdtree.get_node_num()
    assert origin_len == new_len + 1


if __name__ == '__main__':
    test_delete_node()
