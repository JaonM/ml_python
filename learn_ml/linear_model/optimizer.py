"""
Optimizer algorithm
"""
from learn_ml.linear_model import cross_entrophy


def gradient_decent(X, y, n_iter, coef, bias, eta, lr_strategy, penalty, C, alpha, tol, patient):
    """
    Gradient descent optimize method
    :param X: nd-array,with shape (n_sample,n_dim)
    :param y: nd-array,with shape (n_sample,1)
    :param n_iter:
    :param coef:
    :param bias:
    :param eta:
    :param lr_strategy:
    :param penalty:
    :param C:
    :param alpha:
    :param tol:
    :param patient:
    :return:
    """
    act_iter = 0  # actual number of iteration

    # compute initial loss
    last_iter_loss = cross_entrophy(X, y)
    for it in range(n_iter):
        pass
