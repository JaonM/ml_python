import numpy as np


def cross_entropy(y_pred, y_true, normalize=True, eps=1e-15):
    """
    Binary classification cross entropy
    :param y_pred: Array-like with value 0 or 1, shape = (n_sample,)
    :param y_true: Array-like with value 0 or 1, shape = (n_sample,)
    :param normalize: Whether to normalize to mean loss per sample
    :param eps: clip prob p=max(eps,min(1-eps,p))
    :return:
    """
    if type(y_pred) != np.ndarray:
        raise TypeError("Require np.ndarray type,{} checked".format(type(y_pred)))
    if type(y_true) != np.ndarray:
        raise TypeError("Require np.ndarray type,{} checked".format(type(y_true)))
    # clip = np.vectorize(lambda x: max(eps, min(1 - eps, x)))
    # y_pred = clip(y_pred)
    y_pred = np.array(list(map(lambda x: max(eps, min(1 - eps, x)), y_pred)))
    l = np.multiply(y_true, np.log(y_pred)) + np.multiply(1 - y_true, np.log(1 - y_pred))
    loss = -1 * np.sum(l).item()
    if normalize:
        loss = loss / len(y_pred)
    return loss
