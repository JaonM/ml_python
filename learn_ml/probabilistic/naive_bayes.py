"""
Naive Bayes Model
"""
import numpy as np
from learn_ml.utility import save_divide


class NaiveBayes(object):
    """

    Attributes:
        dim:
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            raise ValueError('Not supported data structure')
        self.dim = X.shape[1]
        # save y probability
        self.y_prob = dict()
        for _y, count in np.unique(y, return_counts=True):
            self.y_prob[_y] = count/len(y)
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
                feature_map[column] = p_x_y
            self.feature_map[column] = feature_map

    @staticmethod
    def is_continuous(x, con_ratio=0.95):
        """
        To tell whether a feature is continuous or not
        :param x:
        :con_ration:
        :return:
        """
        _unique = np.unique(x)
        print(type(x[0]))
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
