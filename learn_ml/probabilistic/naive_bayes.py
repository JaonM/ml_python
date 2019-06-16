"""
Naive Bayes Model
"""
import numpy as np
import math
from learn_ml.utility import save_divide


class NaiveBayes(object):
    """

    Parameters:
        _lambda: Laplace smooth lambda value

    Attributes:
        y_prob: Dict like probability object
                Example: {'man':0.23,'woman:0.77}
        x_prob: List dict like probability object
                Example: [{'married':{'man':0.5,'woman':0.5},'unmarried':{'man':0,5,'woman':0.5}},...]
    """

    def __init__(self, _lambda=1):
        self._lambda = _lambda
        self.y_prob = None
        self.x_prob = None

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        if not isinstance(X, np.ndarray):
            raise ValueError('Not supported data structure')
        y_unique = np.unique(y, return_counts=True)
        y_quant = list(map(lambda y: save_divide(y, sum(y_unique[1])), y_unique[1]))
        self.y_prob = dict(zip(y_unique[0], y_quant))

        # iter column in X to compute probability of each feature value
        self.x_prob = []
        for col_idx in range(X.shape[1]):
            if self.is_continuous(X[:, col_idx]):
                mean = np.mean(X[:, col_idx])
                std = np.std(X[:, col_idx])
                prob = {'mean': mean, 'std': std}
            else:
                prob = dict()
                tmp = np.stack((X[:, col_idx], y), axis=1)
                for x in np.unique(X[:, col_idx]):
                    x_y_map = dict()
                    for _y in y_unique[0]:
                        count_x_y = len(tmp[(tmp[:, 0] == x) & (tmp[:, 1]== _y)])
                        p_x_y = save_divide(count_x_y + self._lambda, len(tmp[tmp[:, 1] == _y]) + len(np.unique(
                            X[:, col_idx])) * self._lambda)
                        print(_y, p_x_y,count_x_y)
                        x_y_map[_y] = p_x_y
                    prob[x] = x_y_map
            self.x_prob.append(prob)

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError('Not supported data structure')
        targets = []
        for x in X:
            target, max_p_x_y = None, None
            for y, prob in self.y_prob.items():
                p_x_y = 1
                for i in range(len(x)):
                    if 'mean' in self.x_prob[i]:
                        p_x_y *= self.gaussion_probability_dense(x[i], self.x_prob[i]['mean'], self.x_prob[i]['std'])
                    else:
                        p_x_y *= self.x_prob[i][x[i]][y]
                if not max_p_x_y and not target:
                    target = y
                    max_p_x_y = p_x_y
                else:
                    if p_x_y > max_p_x_y:
                        target = y
                        max_p_x_y = p_x_y
            targets.append(target)
        return np.array(targets)

    @staticmethod
    def gaussion_probability_dense(x, mean, std):
        return 1 / (math.sqrt(2 * math.pi * std ** 2)) * math.exp((-1 * (x - mean) ** 2) / (2 * std ** 2))

    @staticmethod
    def is_continuous(x, con_ratio=0.95):
        """
        To tell whether a feature is continuous or not
        :param x:
        :con_ration:
        :return:
        """
        _unique = np.unique(x)
        if len(_unique) / len(x) >= con_ratio and type(x[0]) != str:
            return True
        else:
            return False


if __name__ == '__main__':
    clf = NaiveBayes()
    x = np.array([[1.1, 'asd', 3],
                  [1.2, 'qqq', 3],
                  [1.3, 'zcxzx', 4]], dtype=object)
    y = np.array([1, 1, 0])
    clf.fit(x, y)
    print(clf.x_prob)
    print(clf.y_prob)
    print(clf.predict(np.array([[1.3, 'zcxzx', 4]], dtype=object)))
