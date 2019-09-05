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
    return 1 / (1 + np.exp(-(np.dot(x, coef) + bias)))


def gradient_decent(X, y, n_iter, coef, bias, lr, lr_strategy, C=None, alpha=None, tol=1e-4, patient=5):
    """
    Gradient descent optimize method
    :param X: nd-array,with shape (n_sample,n_dim)
    :param y: nd-array,with shape (n_sample,)
    :param n_iter:
    :param coef:
    :param bias:
    :param lr:
    :param lr_strategy:
    :param C:
    :param alpha:
    :param tol:
    :param patient: Max times loss no change
    :return:
    """
    act_iter = 0  # actual number of iteration

    # compute initial loss
    y_pred = sigmoid(X, coef, bias)
    best_loss, last_loss = cross_entropy(y_pred, y), cross_entropy(y_pred, y)
    iter_loss_no_change = 0
    losses = []

    for _ in range(n_iter):
        p = np.random.permutation(len(X))
        X, y = X[p], y[p]
        act_iter += 1
        # update coef
        # compute gradient
        h_X = sigmoid(X, coef, bias)
        tmp = np.dot(X.T, h_X - y) / len(X)
        g = lr * tmp
        if C and not alpha:  # l2 penalty
            coef = coef - g - C / len(X) * coef
        elif alpha and not C:  # l1 penalty
            coef = coef - g + alpha / len(X) * (-1 * coef)
        else:
            coef = coef - g
        if bias:
            tmp = h_X - y
            g = lr * np.sum(tmp, axis=0) / len(X)
            bias = bias - g.item()

        y_pred = sigmoid(X, coef, bias)
        cur_loss = cross_entropy(y_pred, y)
        losses.append(cur_loss)
        if cur_loss < last_loss:
            iter_loss_no_change = 0
        last_loss = cur_loss
        if cur_loss <= best_loss:
            best_loss = cur_loss
        if cur_loss > best_loss - tol:
            iter_loss_no_change += 1
            if iter_loss_no_change > patient:
                break

        # TODO learning update strategy
        if lr_strategy != 'fix':
            pass

    return coef, bias, losses, act_iter


def min_batch_gradient_descent(X, y, n_iter, coef, bias, batch_size, lr, lr_strategy, C=None, alpha=None, tol=1e-4,
                               patient=5):
    """
    mini batch gradient descent
    :param X:
    :param y:
    :param n_iter:
    :param coef:
    :param bias:
    :param batch_size:
    :param lr:
    :param lr_strategy:
    :param C:
    :param alpha:
    :param tol:
    :param patient:
    :return:
    """
    act_iter = 0  # actual number of iteration

    # compute initial loss
    y_pred = sigmoid(X, coef, bias)
    best_loss, last_loss = cross_entropy(y_pred, y), cross_entropy(y_pred, y)
    iter_loss_no_change = 0
    losses = []

    for _ in range(n_iter):
        act_iter += 1
        p = np.random.permutation(len(X))
        X_p, y_p = X[p], y[p]
        batch_loss = []
        for X_bat, y_bat in to_batches(X_p, y_p, batch_size):
            # update coef
            # compute gradient
            h_X = sigmoid(X_bat, coef, bias)
            tmp = np.dot(X_bat.T, h_X - y_bat) / len(X_bat)
            g = lr * tmp
            if C and not alpha:  # l2 penalty
                coef = coef - g - C / len(X_bat) * coef
            elif alpha and not C:  # l1 penalty
                coef = coef - g + alpha / len(X_bat) * (-1 * coef)
            else:
                coef = coef - g
            if bias:
                tmp = h_X - y_bat
                g = lr * np.sum(tmp, axis=0) / len(X_bat)
                bias = bias - g.item()
            y_pred = sigmoid(X_bat, coef, bias)
            batch_loss.append(cross_entropy(y_pred, y_bat))

        cur_loss = np.average(batch_loss)
        losses.append(cur_loss)
        if cur_loss < last_loss:
            iter_loss_no_change = 0
        last_loss = cur_loss
        if cur_loss <= best_loss:
            best_loss = cur_loss
        if cur_loss > best_loss - tol:
            iter_loss_no_change += 1
            if iter_loss_no_change > patient:
                break

        # TODO learning update strategy
        if lr_strategy != 'fix':
            pass
    return coef, bias, losses, act_iter


def to_batches(X, y, batch_size):
    for i in range(0, X.shape[0], batch_size):
        yield X[i: i + batch_size, :], y[i: i + batch_size]
