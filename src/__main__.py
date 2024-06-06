import math
import numpy
import matplotlib.pyplot as plt

""" 
    This is the main file of the project.
    This project is an example of the use of the gradient descent algorithm.
    We will use the gradient descent algorithm with a constant learning rate and a optimal learning rate.
    We will use the gradient descent algorithm to find the minimum of the function f(x,y) = x^2 + y^2 +x*y - 3*x - 2*y + 2
"""

def f(x, y):
    return x**2 + y**2 + x*y - 3*x - 2*y + 2

def df(x, y):
    return numpy.array([2*x + y - 3, 2*y + x - 2])

def gradient_descent_constant_learning_rate(x, y, learning_rate, iterations):
    for i in range(iterations):
        x = x - learning_rate * df(x, y)[0]
        y = y - learning_rate * df(x, y)[1]
    return x, y

def gradient_descent_optimal_learning_rate(x, y, iterations):
    for i in range(iterations):
        learning_rate = 1/(i+1)
        x = x - learning_rate * df(x, y)[0]
        y = y - learning_rate * df(x, y)[1]
    return x, y

def main():
    x = 0
    y = 0
    iterations = 1000
    learning_rate = 0.01
    x1, y1 = gradient_descent_constant_learning_rate(x, y, learning_rate, iterations)
    x2, y2 = gradient_descent_optimal_learning_rate(x, y, iterations)
    print("Constant learning rate: ", x1, y1)
    print("Optimal learning rate: ", x2, y2)

    x = numpy.linspace(-10, 10, 100)
    y = numpy.linspace(-10, 10, 100)
    X, Y = numpy.meshgrid(x, y)
    Z = f(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    plt.show()

if __name__ == "__main__":
    main()

