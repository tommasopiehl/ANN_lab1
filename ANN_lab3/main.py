import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def rbf(x, c, sigma):
    """
    Radial basis function
    """
    d = np.sqrt(np.sum(np.square(x - c), axis=0))
    return np.exp(-d ** 2 / (2 * sigma ** 2))


def display_rbf_function():
    n_points = 100
    span = 2
    x = np.linspace(-span, span, n_points)
    y = np.linspace(-span, span, n_points)

    xy = np.zeros((2, n_points ** 2))

    for i in range(len(x)):
        for j in range(len(y)):
            xy[0, i * n_points + j] = x[i]
            xy[1, i * n_points + j] = y[j]

    z = rbf(xy, np.array([1, 1]).reshape(2, -1), 0.5)

    z += rbf(xy, np.array([-1, -1]).reshape(2, -1), 0.5)

    xi, yi = np.meshgrid(x, y)
    zi = griddata((xy[0, :], xy[1, :]), z, (xi, yi), method='cubic')

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(xi, yi, zi, cmap='viridis')  # You can change the colormap as needed

    # Add labels and a color bar
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(surf)

    plt.show()


def generate_gauss_data(n_points, mean, sigma):
    """
    Generates n_points from a gaussian distribution with mean and sigma
    """
    return np.random.normal(mean, sigma, (n_points, 2))


def generate_data(points:list, n_points:list, sigmas:list, classes:list):
    """
    Generates data from a gaussian distributions with mean and sigma
    """
    data = []
    classification = []
    for i in range(len(points)):
        data.append(generate_gauss_data(n_points[i], points[i], sigmas[i]))
        classification.append(np.full(n_points[i], classes[i]))
    return np.concatenate(data), np.concatenate(classification)

# display data distribution
def display_data():
    points = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
    n_points = [100, 100, 100, 100]
    sigmas = [0.5, 0.5, 0.5, 0.5]
    classes = [1, 2, 3, 4]

    data, classification = generate_data(points, n_points, sigmas, classes)

    plt.scatter(data[:, 0], data[:, 1], c=classification)
    plt.show()

