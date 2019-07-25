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


def gradient_decent(X, y, n_iter, coef, bias, eta, lr_strategy, C=None, alpha=None, tol=1e-5, patient=5):
    """
    Gradient descent optimize method
    :param X: nd-array,with shape (n_sample,n_dim)
    :param y: nd-array,with shape (n_sample,)
    :param n_iter:
    :param coef:
    :param bias:
    :param eta:
    :param lr_strategy:
    :param C:
    :param alpha:
    :param tol:
    :param patient: Max times loss no change
    :return:
    """
    act_iter = 0  # actual number of iteration

    # compute initial loss
    y_pred = np.apply_along_axis(sigmoid, 1, X, coef=coef, bias=bias)
    best_loss, last_loss = cross_entropy(y_pred, y), cross_entropy(y_pred, y)
    # h = lambda x: sigmoid(x, coef, bias)
    iter_loss_no_change = 0

    for _ in range(n_iter):
        act_iter += 1
        # update coef
        # compute gradient
        h_X = np.apply_along_axis(sigmoid, 1, X, coef=coef, bias=bias)
        tmp = np.matmul(X.T, h_X - y) / len(X)
        # g = eta * np.sum(tmp, axis=0) / len(X)
        g = eta * tmp
        if C and not alpha:  # l2 penalty
            coef = coef - g - C / len(X) * coef
        elif alpha and not C:  # l1 penalty
            coef = coef - g + alpha / len(X) * (-1 * coef)
        else:
            coef = coef - g
        if bias:
            tmp = h_X - y
            g = eta * np.sum(tmp, axis=0) / len(X)
            bias = bias - g.item()

        y_pred = np.apply_along_axis(sigmoid, 1, X, coef=coef, bias=bias)
        cur_loss = cross_entropy(y_pred, y)
        if cur_loss < last_loss:
            iter_loss_no_change = 0
        last_loss = cur_loss
        if cur_loss <= best_loss:
            best_loss = cur_loss
        if cur_loss > best_loss + tol:
            iter_loss_no_change += 1
            if iter_loss_no_change > patient:
                break

        # TODO learning update strategy
        if lr_strategy != 'fix':
            pass

    return coef, bias, best_loss, act_iter
