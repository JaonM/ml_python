import math

import numpy as np

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
