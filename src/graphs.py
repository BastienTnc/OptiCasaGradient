import matplotlib.pyplot as plt


def plot3DFunction(x, y, z, title, xlabel, ylabel, zlabel):
    """
    Plot a 3D graph.
    :param x: x values (iterable)
    :param y: y values (iterable)
    :param z: z values (iterable)
    :param title: title of the graph (string)
    :param xlabel: label of the x axis (string)
    :param ylabel: label of the y axis (string)
    :param zlabel: label of the z axis (string)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()


def plot3DFunctionAndGradientDescent(x, y, z, fx_values, title, xlabel, ylabel, zlabel,show=True):
    """
    Plot a 3D graph and the points visited by the gradient descent algorithm.
    :param x: x values (iterable)
    :param y: y values (iterable)
    :param z: z values (iterable)
    :param fx_values: points visited by the gradient descent algorithm (iterable of numpy arrays)
    :param title: title of the graph (string)
    :param xlabel: label of the x axis (string)
    :param ylabel: label of the y axis (string)
    :param zlabel: label of the z axis (string)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z)
    ax.plot(fx_values[:, 0], fx_values[:, 1], fx_values[:, 2], color='red', linewidth=5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if show:
        plt.show()
