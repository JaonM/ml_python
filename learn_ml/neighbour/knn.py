"""
k nearest neighbor
"""
import numpy as np
from learn_ml.neighbour.kdtree import KDTree
from learn_ml.utility import save_divide


def get_k_nearest_neighbor(X, k, kdtree, dist_metric='l2'):
    """
    Get X k-nearest node using kd-tree
    :param X: X query data,ndarray
    :param k: number of nearest node
    :param kdtree:
    :param dist_metric: distance metric
    First find the leaf node
    Then backtrack to the root
    :return:
    """
    if kdtree.dim != len(X):
        raise ValueError('Not supported data structure,required ndarray')
    if dist_metric not in ('l2', 'l1'):
        raise ValueError('Not supported distance metric')
    nearest_nodes = []  # keep k nearest node
    backtrack = []  # stack-like list

    # find the leaf node
    node = kdtree.root
    leaf = None
    while node:
        if node.left_child or node.right_child:
            leaf = None
        else:
            leaf = node
        if X[node.cd] <= node.data[node.cd]:
            backtrack.append((node, 'left'))
            node = node.left_child
        else:
            backtrack.append((node, 'right'))
            node = node.right_child
    if leaf:
        radius = dist(leaf.data, X, norm=dist_metric)
        if len(nearest_nodes) < k:
            nearest_nodes.append((leaf, radius))

    # backtrack
    track_pair = backtrack.pop()  # pop the leaf node
    if len(backtrack) > 0:
        track_pair = backtrack.pop()
    else:  # only root node return one result
        return [track_pair[0]]

    while track_pair:
        track_node = track_pair[0]
        d = dist(track_node.data, X, dist_metric)
        if len(nearest_nodes) < k and (track_node, d) not in nearest_nodes:
            nearest_nodes.append((track_node, d))
        else:
            max_pair = max(nearest_nodes, key=lambda x: x[1])
            if max_pair[1] >= d and (track_node, d) not in nearest_nodes:
                nearest_nodes.remove(max_pair)
                nearest_nodes.append((track_node, d))
        pair = max(nearest_nodes, key=lambda x: x[1])
        # whether to search the subtree of track node
        if abs(track_node.data[track_node.cd] - pair[0].data[track_node.cd]) < pair[1] \
                or len(nearest_nodes) < k:
            # search subtree
            if track_node.left_child or track_node.right_child:
                if track_pair[1] == 'left' and track_node.right_child:  # search right subtree
                    node = track_node.right_child
                elif track_pair[1] == 'right' and track_node.left_child:
                    node = track_node.left_child
                else:
                    node = None
                _leaf = None
                while node:
                    if node.left_child or node.right_child:
                        _leaf = None
                    else:
                        _leaf = node
                    if X[node.cd] <= node.data[node.cd]:
                        backtrack.append((node, 'left'))
                        node = node.left_child
                    else:
                        backtrack.append((node, 'right'))
                        node = node.right_child
            else:
                _leaf = track_node
            if _leaf:
                d = dist(_leaf.data, X, dist_metric)
                if len(nearest_nodes) < k and (track_node, d) not in nearest_nodes:
                    nearest_nodes.append((_leaf, d))
                else:
                    max_pair = max(nearest_nodes, key=lambda x: x[1])
                    if max_pair[1] >= d and (track_node, d) not in nearest_nodes:
                        nearest_nodes.remove(max_pair)
                        nearest_nodes.append((_leaf, d))
        if len(backtrack) > 0:
            track_pair = backtrack.pop()
        else:
            track_pair = None
    return list(map(lambda x: x[0], sorted(nearest_nodes, key=lambda x: x[1])))


class KNN(object):
    """
    K-Nearest Neighbor Model

    Attributes:
        candidates: Candidates equivalent to training set,nd-array shape=(n_sample,n_dim)
        labels: Categorical labels set, can be
        kdtree: Built kd-tree
        k: number of neighbor to return
        dist_metric: distance measurement method
    """

    def __init__(self, candidates, labels, k, dist_metric='l2'):
        self.candidates = candidates
        self.labels = labels
        self.k = k
        self.kdtree = KDTree.create(candidates.shape[1], candidates, labels=labels)
        self.dist_metric = dist_metric

    def predict(self, X, regression=False):
        """
        Predict function
        :param X: nd-array
        :return: label
        """
        nodes = get_k_nearest_neighbor(X, self.k, self.kdtree, self.dist_metric)
        label_count = dict()
        if regression:
            _sum = 0
            for n in nodes:
                _sum += n.label
            return save_divide(_sum, len(nodes))
        else:
            for n in nodes:
                label_count[n.label] = label_count.get(n.label, 0) + 1
            labels = sorted(label_count.items(), key=lambda x: x[1], reverse=True)
            return labels[0][0]


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
        return np.linalg.norm(x1 - x2, ord=None)
    elif norm == 'l1':
        return np.linalg.norm(x1 - x2, ord=1)
    return 0


if __name__ == '__main__':
    candidates = np.random.rand(4, 3)
    print(candidates)
    kdtree = KDTree.create(3, candidates, np.zeros(len(candidates)))
    knn = KNN(candidates, np.zeros(len(candidates)), 3, 'l2')
    print(knn.predict(candidates[0]))
