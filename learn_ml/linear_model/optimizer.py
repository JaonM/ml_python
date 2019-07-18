"""
Optimizer algorithm
"""
import math

import numpy as np

from learn_ml.metrics import cross_entropy


def sigmoid(x, coef=None, bias=None):
    if type(x) != np.ndarray:
        return 1 / (1 + math.exp(-x))
    if coef is None:
        coef = np.array([1] * len(x))
    if bias is None:
        bias = 0
    return 1 / (1 + np.exp(-(np.dot(coef, x) + bias)))


def gradient_decent(X, y, n_iter, coef, bias, eta, lr_strategy, penalty, C, alpha, tol, patient):
    """
    Gradient descent optimize method
    :param X: nd-array,with shape (n_sample,n_dim)
    :param y: nd-array,with shape (n_sample,1)
    :param n_iter:
    :param coef:
    :param bias:
    :param eta:
    :param lr_strategy:
    :param penalty:
    :param C:
    :param alpha:
    :param tol:
    :param patient:
    :return:
    """
    act_iter = 0  # actual number of iteration

    # compute initial loss
    last_iter_loss = cross_entropy(y_pred, y_true)
    h = np.vectorize(lambda x: sigmoid(x, coef, bias))

    for _ in range(n_iter):
        act_iter += 1
        # update coef
        # compute gradient
        tmp = (h(X) - y) * X
        g = eta * np.sum(tmp, axis=0) / len(X)
        if C and not alpha:  # l2 penalty
            coef = coef - g - C / len(coef) * coef
        elif alpha and not C:  # l1 penalty
            coef = coef - g + alpha / len(coef) * (-1 * coef)
        else:
            coef = coef - g

        # update bias
        if bias:
            tmp = h(X - y)
            g = eta * np.sum(tmp, axis=0) / len(X)
            bias = bias - g.item()

        cur_loss = cross_entropy()
