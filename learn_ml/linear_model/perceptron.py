"""
感知机学习算法
"""
import numpy as np


class Perceptron(object):
    """
    Remaining Problem：
        Training sometimes won't be convergence

    Attributes:
       eta : float
            Learning rate
       epoch : int
            Number of training iteration
       shuffle : boolean
            Whether to shuffle the data
       weight : array, shape=(n_features,)
            Perceptron training weight
       bias : float
            Perceptron training bias

    """

    def __init__(self, shuffle=True, eta=0.01, epoch=1):
        self.eta = eta
        self.epoch = epoch
        self.shuffle = shuffle

    def _decision_function(self, X):
        """
        决策函数 y=w*x+b
        :param X: nd-array (n_samples,n_features)
        :return: (n_samples,1)
        """
        return np.matmul(X, np.transpose(self.weight)) + self.bias

    def loss(self, X, y):
        """
        损失函数
        :param X: nd-array (n_samples,n_features)
        :param y: nd-array (n_samples,)
        :return:
        """
        y = y.reshape(-1, 1)
        l = np.multiply(y, self._decision_function(X))
        l = l.flatten()
        l = l[l <= 0]
        return -1 * np.sum(l)

    def fit(self, X, y):
        """
        训练数据
        :param X: nd-array (n_samples,n_features)
        :param y: nd-array (n_samples,)
        :return:
        """
        self.weight = np.random.rand(X.shape[1])
        self.bias = np.random.random()

        if self.shuffle:
            np.random.shuffle(X)
        for _epoch in range(self.epoch):
            correct_clf_count = 0
            err_clf_count = 0
            print('training {} epoch'.format(_epoch))
            w_delta = np.zeros(self.w.shape)
            b_delta = 0
            for i in range(X.shape[0]):
                _y = self._decision_function(X[i]).item() * y[i]
                if _y <= 0:
                    # update argument
                    self.weight += np.multiply(self.eta, np.multiply(y[i], X[i]))
                    self.bias += self.eta * y[i]
                    w_delta += np.multiply(y[i], X[i])
                    b_delta += y[i]
                    err_clf_count += 1
                else:
                    correct_clf_count += 1
            # self.w += np.multiply(self.eta, w_delta / err_clf_count)
            # self.b += self.eta * b_delta / err_clf_count
            print(correct_clf_count)
            if correct_clf_count == X.shape[0]:
                break

    def predict(self, X):
        res = self._decision_function(X).flatten()
        for i in range(len(res)):
            if res[i] > 0:
                res[i] = 1
            else:
                res[i] = -1
        return res

    def get_param(self):
        return self.weight, self.bias
