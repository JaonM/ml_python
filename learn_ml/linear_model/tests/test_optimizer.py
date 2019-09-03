import math

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

from ..optimizer import gradient_decent, min_batch_gradient_descent
from ..optimizer import sigmoid, to_batches


def test_sigmoid():
    x = 0
    y = sigmoid(x)
    assert y == 0.5

    x = np.array([0, 0, 0, 0])
    y = sigmoid(x)
    assert y == 0.5

    x = np.array([1, 1, 1, 1])
    y = sigmoid(x)
    print(y)
    assert y == 1 / (1 + math.exp(-4))

    x = np.array([1, 1, 1, 1])
    coef = np.array([0.5, 0.5, 1, 1])
    y = sigmoid(x, coef)
    print(y)
    assert y == 1 / (1 + math.exp(-3))

    x = np.array([1, 1, 1, 1])
    coef = np.array([0.5, 0.5, 1, 1])
    bias = 1
    y = sigmoid(x, coef, bias)
    print(y)
    assert y == 1 / (1 + math.exp(-4))


def test_gradient_decent():
    X, y = load_breast_cancer(return_X_y=True)

    coef = np.random.rand(X.shape[1])
    bias = np.random.rand(1).item()
    # bias = None
    n_iter = 1000
    eta = 0.1
    coef, bias, loss, act_iter = gradient_decent(X, y, n_iter, coef, bias, eta, 'fix')

    print(act_iter)
    y_pred = np.apply_along_axis(sigmoid, 1, X, coef, bias)
    y_pred = list(map(lambda x: binary(x), y_pred))
    score = accuracy_score(y, y_pred)
    print(score)


def test_mini_gradient_descent():
    X, y = load_breast_cancer(return_X_y=True)

    coef = np.random.rand(X.shape[1])
    bias = np.random.rand(1).item()
    # bias = None
    n_iter = 1000
    eta = 0.1
    batch_size = 12
    print(X.shape, y.shape)
    coef, bias, loss, act_iter = min_batch_gradient_descent(X, y, n_iter, coef, bias, batch_size, eta, 'fix')

    print(act_iter)
    y_pred = np.apply_along_axis(sigmoid, 1, X, coef, bias)
    y_pred = list(map(lambda x: binary(x), y_pred))
    score = accuracy_score(y, y_pred)
    print(score)


def test_to_batches():
    X, y = np.random.rand(1000, 10), np.random.rand(1000)
    batch_size = 10
    for _x, _y in to_batches(X, y, batch_size):
        assert _x.shape == (10, 10)
        assert _y.shape == (10,)


def binary(x):
    if x >= 0.5:
        return 1
    else:
        return 0
