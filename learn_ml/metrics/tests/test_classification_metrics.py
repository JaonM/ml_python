import numpy as np
from sklearn.metrics import log_loss

from ..classfication import cross_entropy


def test_cross_entropy():
    y_pred, y_true = np.array([0.9, 0.1, 0.1, 0.9]), np.array([1, 0, 0, 1])
    loss = cross_entropy(y_pred, y_true)
    sk_loss = log_loss(y_true, y_pred)

    assert loss == sk_loss
