"""
逻辑回归模型

TODO
    1. 支持不同的优化器
    2. 支持多种正则化项
    3. 支持并行
"""
import numpy as np
import pandas as pd


class LogisticsRegression(object):
    """
    Logistics Regression Model

    Parameters:
        max_iter : int,Max number of iteration,default 100
        penalty : str,'l2','l1' regularization,default None
        optimizer: str,'gd','sgd','min-sgd','new-ton',...
        tol: float,Tolerance for stopping criteria,default TODO
        shuffle: Whether shuffle the data,default True
        eta: Learning rate
        lr_strategy: 'fix','auto decay'
        C: L2 regularization parameter,default None
        alpha: L1 regularization parameter,default None
        with_bias: Whether contains bias,default False
        patient: Number of iteration that no changing loss

    Attributes:
        classes_: Class list
        bias: Bias if with_bias is True,default None
        coef: Model coefficient
        n_iter: Number of iteration that actual fits
    """

    def __int__(self, max_iter=100, penalty=None, optimizer='gd', tol=1e-4, shuffle=True, eta=1e-3,
                lr_strategy='fix', C=None, alpha=None, with_bias=False, patient=5):
        self.max_iter = max_iter
        self.penalty = penalty
        self.optimizer = optimizer
        self.tol = tol
        self.shuffle = shuffle
        self.eta = eta
        self.lr_strategy = lr_strategy
        self.C = C
        self.alpha = alpha
        self.with_bias = with_bias
        self.patient=patient

    def fit(self, X, y):
        """

        :param X: array-like
        :param y: array-like
        :return:
        """
        X, y = self._check_input(X), self._check_input(y)
        self._check_penalty()

        if shuffle:
            np.random.shuffle(X)

        self.coef, self.bias = self._init_coefficient(X)

        self._init_parameters()

        if self.optimizer == 'gd':
            pass

    def _init_parameters(self):
        if self.penalty:
            if self.penalty == 'l2':
                self.C = 1
            elif self.penalty == 'l1':
                self.alpha = 1
            else:
                self.C, self.alpha = 1, 1

    def _init_coefficient(self, x):
        # TODO 可能要加入初始化策略
        coef = np.random.rand(x.shape[1])
        if self.with_bias:
            bias = np.random.randn()
        else:
            bias = None
        return coef, bias

    @staticmethod
    def _check_input(x):
        """
        Check the type of input and change it into nd-array
        :param x:
        :return:
        """
        if type(x) == np.ndarray and len(x.shape) > 1:
            return x
        elif type(x) == np.ndarray and len(x.shape) == 1:
            return x.reshape(-1, 1)
        elif type(x) == list and len(np.array(x).shape) > 1:
            return np.array(x)
        elif type(x) == pd.DataFrame and len(x.shape) > 1:
            return x.to_numpy()
        else:
            raise TypeError("Not support input type")

    def _check_penalty(self):
        if self.penalty and (not self.C or not self.alpha):
            raise ValueError("Not valid penalty setting")
        elif self.penalty == 'l2' and not self.C:
            raise ValueError("Not valid penalty setting")
        elif self.penalty == 'l1' and not self.alpha:
            raise ValueError("Not valid penalty setting")
