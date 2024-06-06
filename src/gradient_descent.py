# -*- coding: utf-8 -*-
import numpy as np


def gradientDescentConstantStepSize(f, df, x0, step_size, tol=1e-6, max_iter=1000, return_steps=False):
    """
    Apply gradient descent with a constant step size to minimize the function f.
    :param f: function to minimize
    :param df: gradient of the function to minimize
    :param x0: initial point (numpy array)
    :param step_size: step size (float)
    :param tol: tolerance for convergence (float)
    :param max_iter: maximum number of iterations (int)
    :return: the points visited by the algorithm (list of numpy arrays)
    """
    x_values = [x0]
    for i in range(max_iter):
        x_values.append(x_values[-1] - step_size * df(x_values[-1]))
        if np.linalg.norm(x_values[-1] - x_values[-2]) < tol:
            break
    return x_values if return_steps else x_values[-1]


def gradientDescentOptimalStepSize(f, df, x0, tol=1e-6, max_iter=1000, return_steps=False):
    """
    Apply gradient descent with an optimal step size to minimize the function f.
    :param f: function to minimize
    :param df: gradient of the function to minimize
    :param x0: initial point (numpy array)
    :param tol: tolerance for convergence (float)
    :param max_iter: maximum number of iterations (int)
    :return: the point that minimizes the function (numpy array)
    """
    x_values = [x0]
    for i in range(max_iter):
        step_size = 1 / (i + 1)
        x_values.append(x_values[-1] - step_size * df(x_values[-1]))
        if np.linalg.norm(x_values[-1] - x_values[-2]) < tol:
            break
    return x_values if return_steps else x_values[-1]
