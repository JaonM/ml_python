"""
k nearest neighbor
"""


def get_k_nearest_neighbor(X, k, kdtree):
    """
    :param X: X query data,ndarray
    :param k: number of nearest node
    :param kdtree:
    First find the leaf node
    Then backtrack to the root
    :return:
    """
    if kdtree.dim != X.shape[1]:
        raise ValueError('Not supported data structure,required ndarray')
    pass
