"""
knn tests script
"""
import numpy as np
from learn_ml.neighbour.knn import KNN, DecisionEnum


def test_knn():
    candidates = np.random.rand(10, 3)
    labels = np.random.randint(0, 5, size=10)
    knn = KNN(candidates, labels=labels, k=3, dist_metric='l2')
    query = np.random.rand(3)
    r = knn.predict(query)
    assert r in [0, 1, 2, 3, 4]

    r = knn.predict(query, decision_strategy=DecisionEnum.NEAREST)
    assert r in [0.0, 1.0, 2.0, 3.0, 4.0]

    r = knn.predict(query, regression=True)
    assert r not in [0, 1, 2, 3, 4]

    r = knn.predict(query, regression=True, decision_strategy=DecisionEnum.NEAREST)
    assert r in [0, 1, 2, 3, 4]
