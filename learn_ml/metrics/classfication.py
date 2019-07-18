import numpy as np


def cross_entropy(y_pred, y_true, normalize=True):
    """
    Binary classification cross entropy
    :param y_pred: Array-like with value 0 or 1
    :param y_true: Array-like with value 0 or 1
    :param normalize: Whether to normalize to mean loss per sample
    :return:
    """
    if type(y_pred) != np.ndarray:
        raise TypeError("Require np.ndarray type,{} checked".format(type(y_pred)))
    if type(y_true) != np.ndarray:
        raise TypeError("Require np.ndarray type,{} checked".format(type(y_true)))
    l = np.multiply(y_true, np.log(y_pred)) + np.multiply(1 - y_true, np.log(1 - y_pred))
    loss = -1 * np.sum(l).item()
    if normalize:
        loss = loss / len(y_pred)
    return loss
