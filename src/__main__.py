import numpy
import sys
from gradient_descent import gradientDescentConstantStepSize, gradientDescentOptimalStepSize
from graphs import plot3DFunctionAndGradientDescent
from example import f as example_f, df as example_df

def f(X):
    x = X[0]
    y = X[1]
    return x**2 + y**2 + x*y - 3*x - 2*y + 2


def df(X):
    x = X[0]
    y = X[1]
    return numpy.array([2*x + y - 3, 2*y + x - 2])


X = numpy.meshgrid(numpy.linspace(-10, 10, 100), numpy.linspace(-10, 10, 100))


if 'example' in sys.argv:
    Z = example_f(X)
    x0 = numpy.array([-8,-8])
    X_values = gradientDescentConstantStepSize(example_f, example_df, x0, 0.1, return_steps=True)
    fx_values = numpy.array([[x_value[0], x_value[1], example_f(x_value)] for x_value in X_values])
    plot3DFunctionAndGradientDescent(X[0], X[1], Z, fx_values, 'f(x, y) = amplitude * exp(-((x - x0)^2 / (2 * sigma_x^2) + (y - y0)^2 / (2 * sigma_y^2)))', 'x', 'y', 'f(x, y)')
else:
    Z = f(X)
    x0 = numpy.array([8, 8])
    X_values = gradientDescentConstantStepSize(f, df, x0, 0.1, return_steps=True)
    fx_values = numpy.array([[x_value[0], x_value[1], f(x_value)] for x_value in X_values])

    plot3DFunctionAndGradientDescent(X[0], X[1], Z, fx_values, 'f(x, y) = x^2 + y^2 + xy - 3x - 2y + 2', 'x', 'y', 'f(x, y)')
