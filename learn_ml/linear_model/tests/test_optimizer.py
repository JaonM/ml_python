import math

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

from ..optimizer import gradient_decent
from ..optimizer import sigmoid


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
    # X = np.array([[1], [0], [0], [1], [1]])
    # y = np.array([1, 0, 0, 1, 1])
    # coef = np.array([1])
    # X = np.random.rand(10, 1)
    # y = np.random.randint(0, 2, size=10)

    X, y = load_breast_cancer(return_X_y=True)

    coef = np.random.rand(X.shape[1])
    bias = np.random.rand(1).item()
    # bias = None
    n_iter = 2000
    eta = 0.1
    coef, bias, loss, act_iter = gradient_decent(X, y, n_iter, coef, bias, eta, 'fix')

    print(act_iter)
    y_pred = np.apply_along_axis(sigmoid, 1, X, coef, bias)
    y_pred = list(map(lambda x: binary(x), y_pred))
    score = accuracy_score(y, y_pred)
    print(score)

    assert 1 == 0


def binary(x):
    if x >= 0.5:
        return 1
    else:
        return 0