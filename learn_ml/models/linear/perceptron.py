"""
感知机学习算法
"""
import numpy as np


class Perceptron(object):
    """
    W in shape (n_feature,)
    b scalar
    """

    def __init__(self, shuffle=True, eta=0.01, epoch=1):
        self.optimizer = 'sgd'
        self.eta = eta
        self.epoch = epoch
        self.shuffle = shuffle

    def _decision_function(self, X):
        """
        决策函数 y=w*x+b
        :param X: nd-array (n_sample,n_feature)
        :return: (n_sample,1)
        """
        return np.matmul(X, np.transpose(self.w)) + self.b

    def loss(self, X, y):
        """
        损失函数
        :param X: nd-array (n_sample,n_feature)
        :param y: nd-array (n_sample,)
        :return:
        """
        y = y.reshape(-1, 1)
        l = np.multiply(y, self._decision_function(X))
        l = l.flattern()
        l = l[l <= 0]
        return -1 * np.sum(l)

    def fit(self, X, y):
        """
        训练数据
        :param X:
        :param y:
        :return:
        """
        self.w = np.random.rand(X.shape[1])
        self.b = np.random.random()

        if self.shuffle:
            np.random.shuffle(X)
        for _epoch in range(self.epoch):
            correct_clf_count = 0
            print('training {} epoch'.format(_epoch))
            for i in range(X.shape[0]):
                _y = self._decision_function(X[i]).item() * y[i]
                if _y <= 0:
                    # update argument
                    self.w = self.w + np.multiply(self.eta * y[i], X[i])
                    self.b = self.b + self.eta * y[i]
                else:
                    correct_clf_count += 1
            if correct_clf_count == X.shape[0]:
                break

    def predict(self, X):
        return 1 if self._decision_function(X) > 0 else -1

    def _get_param(self):
        return self.w, self.b


if __name__ == '__main__':
    X = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])
    # X = np.random.rand(100, 2)
    # y = np.random.randint(0, 2, 100)
    # for i in range(len(y)):
    #     if y[i] == 0:
    #         y[i] = -1

    clf = Perceptron(epoch=10000, eta=0.1)
    clf.fit(X, y)
    print(clf.predict([[5, 5]]))
