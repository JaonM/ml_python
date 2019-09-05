"""
逻辑回归模型

TODO
    1. 支持不同的优化器
    2. 支持多种正则化项
    3. 支持并行
    增加gd拟合测试

"""
import pandas as pd

from .optimizer import *
from sklearn.linear_model import SGDClassifier

class LogisticsRegression(object):
    """
    Logistics Regression Model

    Parameters:
        max_iter : int,Max number of iteration,default 100
        penalty : str,'l2','l1' regularization,default None
        optimizer: str,'gd','sgd','min-sgd','new-ton',...
        tol: float,Tolerance for stopping criteria,default TODO
        shuffle: Whether shuffle the data,default True
        lr: Learning rate
        lr_strategy: 'fix','auto decay'
        C: L2 regularization parameter,default None
        alpha: L1 regularization parameter,default None
        with_bias: Whether contains bias,default False
        patient: Number of consecutive iteration that not improving the training loss
        batch_size: Number of samples per batch

    Attributes:
        classes_: Class list
        bias: Bias if with_bias is True,default None
        coef: Model coefficient
        n_iter: Number of iteration that actual fits
        loss: Value of loss function
    """

    def __init__(self, max_iter=100, penalty=None, optimizer='sgd', tol=1e-4, shuffle=True, lr=0.1,
                 lr_strategy='fix', C=None, alpha=None, with_bias=False, patient=5, batch_size=32):
        self.max_iter = max_iter
        self.penalty = penalty
        self.optimizer = optimizer
        self.tol = tol
        self.shuffle = shuffle
        self.lr = lr
        self.lr_strategy = lr_strategy
        self.C = C
        self.alpha = alpha
        self.with_bias = with_bias
        self.patient = patient
        self.n_iter = 0
        self.losses = None
        self.batch_size = batch_size

    def fit(self, X, y):
        """

        :param X: array-like
        :param y: array-like
        :return:
        """
        X, y = self._check_input(X), self._check_input(y)
        self._check_penalty()

        if self.shuffle:
            p = np.random.permutation(len(X))
            X, y = X[p], y[p]

        self.coef, self.bias = self._init_coefficient(X)

        self._init_penalty()

        if self.optimizer == 'gd':
            self.coef, self.bias, self.losses, self.n_iter = gradient_decent(X, y, self.max_iter, self.coef,
                                                                             self.bias, self.lr, self.lr_strategy,
                                                                             self.C, self.alpha,
                                                                             self.tol, self.patient)
        elif self.optimizer == 'sgd':
            self.coef, self.bias, self.losses, self.n_iter = min_batch_gradient_descent(X, y, self.max_iter, self.coef,
                                                                                        self.bias, self.batch_size,
                                                                                        self.lr, self.lr_strategy,
                                                                                        self.C, self.alpha, self.tol,
                                                                                        self.patient)

    def predict(self, X, theta=0.5):
        def binary(x):
            if x >= theta:
                return 1
            else:
                return 0

        r = np.array(list(map(binary, self.predict_proba(X))))
        return r

    def predict_proba(self, X):
        """

        :param X: array-like with shape(n_sample,n_dim)
        :return:
        """
        return np.apply_along_axis(sigmoid, 1, X, coef=self.coef, bias=self.bias)

    def _init_penalty(self):
        if self.penalty:
            if self.penalty == 'l2' and not self.C:
                self.C = 1
            elif self.penalty == 'l1' and not self.alpha:
                self.alpha = 0.1
            else:
                self.C, self.alpha = None, None

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
        if type(x) == np.ndarray and len(x.shape) >= 1:
            return x

        elif type(x) == list and len(np.array(x).shape) > 1:
            return np.array(x)
        elif type(x) == pd.DataFrame and len(x.shape) > 1:
            return x.to_numpy()
        else:
            raise TypeError("Not support input type")

    def _check_penalty(self):
        if self.penalty and not self.C and not self.alpha:
            raise ValueError("Not valid penalty setting")
        elif self.penalty == 'l2' and not self.C:
            raise ValueError("Not valid penalty setting")
        elif self.penalty == 'l1' and not self.alpha:
            raise ValueError("Not valid penalty setting")
