"""
knn test script
"""
import numpy as np
from learn_ml.neighbour.knn import KNN


def test_knn():
    candidates = np.random.rand(10, 3)
    labels = np.random.randint(0, 5, size=10)
    knn = KNN(candidates, labels=labels, k=3, dist_metric='l2')
    query = np.random.rand(3)
    r = knn.predict(query)
    assert r in [0, 1, 2, 3, 4]
