import numpy as np


A = 2
B = 0.5
C = 1
D = 3
E = 4
F = 2

def f(X):
    return A * X[0]**2 + B * X[1]**2 + C * X[0] * X[1] - D * X[0] + E* X[1] + F


def df(X):
    return np.array([2 * A * X[0] + C * X[1] - D, 2 * B * X[1] + C * X[0] - E])
