"""
k nearest neighbor
"""
import numpy as np


def get_k_nearest_neighbor(X, k, kdtree, dist_metric='l2'):
    """
    :param X: X query data,ndarray
    :param k: number of nearest node
    :param kdtree:
    :param dist_metric: distance metrice
    First find the leaf node
    Then backtrack to the root
    :return:
    """
    if kdtree.dim != X.shape[1]:
        raise ValueError('Not supported data structure,required ndarray')
    if dist_metric not in ('l2', 'l1'):
        raise ValueError('Not supported distance metric')
    nearest_nodes = []  # keep k nearest node
    backtrack = []  # stack-like list

    # find the leaf node
    node = kdtree.root
    leaf = node
    while node:
        if X[node.cd] <= node[node.cd]:
            node = node.left_child
            backtrack.append((node, 'left'))
            leaf = node
        else:
            node = node.right_child
            backtrack.append((node, 'right'))
            leaf = node
    radius = dist(leaf.data, X, norm=dist_metric)
    if len(nearest_nodes) <= k:
        nearest_nodes.append((leaf, radius))

    # backtrack
    track_pair = backtrack.pop()
    while track_pair:
        track_node = track_pair[0]
        pair = max(nearest_nodes, key=lambda x: x[1])
        # whether to search the subtree of track node
        if abs(track_node.data[track_node.cd] - pair[0].data[track_node.cd]) < pair[1]:
            # search subtree
            if track_pair[1] == 'left' and track_node.right_child:  # search right subtree
                pass
            elif track_pair[1] == 'right' and track_node.left_child:
                pass
            else:
                pass


def dist(x1, x2, norm='l2'):
    """
    Calculate distance between two nodes
    :param x1:
    :param x2:
    :param norm: order of norm
    :return:
    """
    if norm not in ('l2', 'l1'):
        raise ValueError('Not supported distance metric')
    if norm == 'l2':
        return np.linalg.norm(x1, x2, norm=2)
    elif norm == 'l1':
        return np.linalg.norm(x1, x2, norm=1)
    return 0
