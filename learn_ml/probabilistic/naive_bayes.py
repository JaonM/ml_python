"""
Naive Bayes Model
"""
import numpy as np
import math
from learn_ml.utility import save_divide


class NaiveBayes(object):
    """

    Attributes:
        dim:
        y_prob: Dict like probability object
                Example: {'man':0.23,'woman:0.77}
        x_prob: List dict like probability object
                Example: [{'married':{'man':0.5,'woman':0.5},'unmarried':{'man':0,5,'woman':0.5}},...]
    """

    def __init__(self, smooth_strategy='laplace'):
        self.smooth_strategy = smooth_strategy
        self.y_prob = None
        self.x_prob = None
        self.dim = None

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        if not isinstance(X, np.ndarray):
            raise ValueError('Not supported data structure')
        self.dim = X.shape[1]
        y_unique = np.unique(y, return_counts=True)
        y_quant = list(map(lambda y: save_divide(y, len(y)), y_unique[1]))
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
                    for y in y_unique[0]:
                        count_x_y = len(tmp[(tmp[:, 0] == x) & (tmp[:, 1]) == y])
                        if count_x_y == 0:
                            # TODO add smooth method
                            p_x_y = 0.0001
                        else:
                            p_x_y = count_x_y / len(tmp[tmp[:, 1] == y])
                        x_y_map[y] = p_x_y
                    prob[x] = x_y_map
            self.x_prob.append(prob)

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError('Not supported data structure')
        for x in X:
            prob = 1
            for y,prob in self.y_prob.items():
                for i in range(len(x)):
                    p_x_y=0
                    x_y_map = self.x_prob[i]



    def fit_backup(self, X, y):
        if not isinstance(X, np.ndarray):
            raise ValueError('Not supported data structure')
        self.X = X
        self.y = y
        self.dim = X.shape[1]
        # save y probability
        self.y_prob = dict()
        for _y, count in np.unique(y, return_counts=True):
            self.y_prob[_y] = count / len(y)
        self.feature_map = dict()
        for column in range(X.shape[1]):
            feature_map = dict()
            if self.is_continuous(X[:, column]):
                feature_map['type'] = 'continuous'
                feature_map['mean'] = np.mean(X[:, column])
                feature_map['std'] = np.std(X[:, column])
            else:
                feature_map['type'] = 'discrete'
                p_x_y = dict()
                for i in range(len(X[:, column])):
                    if X[:, column][i] in p_x_y.keys():
                        p_x_y[X[:, column][i]][y[i]] = p_x_y[X[:, column][i]].get(y[i], 0) + 1
                    else:
                        p_x_y[X[:, column][i]] = dict()
                        p_x_y[X[:, column][i]][y[i]] = p_x_y[X[:, column][i]].get(y[i], 0) + 1
                # convert to prob
                for key, value in p_x_y.items():
                    for _key, _value in value.items():
                        p_x_y[key][_key] = _value / len(y)
                feature_map['prob'] = p_x_y
            self.feature_map[column] = feature_map

    @staticmethod
    def gaussion_probality_dense(x, mean, std):
        return 1 / (math.sqrt(2 * math.pi * std ** 2)) * math.exp((-1 * (x - mean) ** 2) / (2 * std ** 2))

    def predict_backup(self, X):
        """

        :param X: require 2-d array
        :return:
        """
        if X.shape[1] != self.dim:
            raise ValueError('Data structure error,require 2-d array')
        ret = []
        for sample in X:
            target, max_p = None, None
            for y in self.y_prob.keys():
                p = 1
                for i in range(self.dim):
                    if self.feature_map[i]['type'] == 'continuous':
                        p = p * self.gaussion_probality_dense(sample[i], self.feature_map[i]['mean'],
                                                              self.feature_map[i]['std'])
                    else:
                        # TODO add smooth method
                        if y in self.feature_map[i]['prob'][sample[i]].keys():
                            p = p * self.feature_map[i]['prob'][sample[i]][y]
                        else:
                            # laplace smooth
                            p = p * (self.feature_map[i]['prob'][sample[i]][y] * len(self.y) + 1 / len(self.y) + len(
                                self.feature_map[i]['prob'].keys()))
                            # p=p*0.001
                if not target and not max_p:
                    target = y
                    max_p = p
                elif p > max_p:
                    max_p = p
                    target = y
            ret.append(target)
        return ret

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
    print(clf.feature_map)
    print(clf.y_prob)
    print(clf.predict(np.array([[1.3, 'zcxzx', 4]], dtype=object)))
