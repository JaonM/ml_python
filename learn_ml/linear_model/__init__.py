import numpy as np

from learn_ml.linear_model.perceptron import Perceptron


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cross_entrophy(x, y):
    pass
