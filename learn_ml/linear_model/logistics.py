"""
逻辑回归模型

TODO
    1. 支持不同的优化器
    2. 支持多种正则化项
    3. 支持多分类
    4. 支持并行
"""


class LogisticsRegression(object):
    """
    Logistics Regression Model

    Parameters:
        max_iter : Max number of iteration,default 100
        penalty : 'l2','l1' regularization,default None
        optimizer: 'gd','sgd','new-ton',...
        tol: Tolerance for stopping criteria
        shuffle: Whether shuffle the data,default True
        eta: Learning rate
        lr_strategy: 'fix','auto decay'
        is_multiclass: Whether fit the multi-class model,default False
        C: L2 regularization parameter,default None
        alpha: L1 regularization paramter,default None
        with_bias: Whether contains bias,default False

    Attributes:
        classes_: Class list
        bias:
        coeff: Model coefficient
        n_iter: Number of iteration that actual fits
    """

    def __int__(self, max_iter):
        pass
