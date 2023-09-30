import numpy as np
from matplotlib import pyplot as plt

from ANN_lab2.animator_SOM import Animator


def egg_carton(x, y):
    return np.sin(0.3 * x) + np.cos(0.1 * y)


def plane(x, y):
    return x + y


def u_shape(x, y):
    a = np.log(x + 2.1)
    c = np.min(a)
    a = a - c
    # generate -1 or 1 with equal probability in the same shape as x
    b = np.random.choice([-1, 1], size=x.shape)

    return a * b


def get_surface_dots(n_points=1000, x_min=-10, x_max=20, y_min=-5, y_max=60, func=None, noise=0.0):
    if func is None:
        func = egg_carton

    # Generate points in the x,y space from uniform distribution

    x = np.random.uniform(x_min, x_max, n_points)
    y = np.random.uniform(y_min, y_max, n_points)

    # Generate points in the z space from the function
    z = func(x, y)

    if noise > 0:
        z += np.random.normal(0, noise, n_points)

    return np.array([x, y, z])


# display data distribution
def display_data():
    data = get_surface_dots()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.scatter(data[0, :], data[1, :], data[2, :])
    plt.show()


class SOM:

    def __init__(self, shape, dimension, random=True):
        self.shape = shape
        # instantiate centers in a grid
        self.centers = np.zeros(shape + (dimension,))
        if random:
            self.centers = np.random.uniform(0, 1, shape + (dimension,))
        else:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    self.centers[i, j, 0] = i
                    self.centers[i, j, 1] = j

    def get_winner(self, x):
        """
        Returns the indeces of the closest center to x
        """
        ix, jx = 0, 0
        min_dist = np.linalg.norm(x - self.centers[0, 0, :])

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                dist = np.linalg.norm(x - self.centers[i, j, :])
                if dist < min_dist:
                    min_dist = dist
                    ix, jx = i, j

        return ix, jx

    def update(self, x, eta, sigma):
        """
        Updates the centers according to the SOM update rule
        """
        i, j = self.get_winner(x)

        # update the centers
        for k in range(self.shape[0]):
            for l in range(self.shape[1]):
                grid_distance = np.sqrt((i - k) ** 2 + (j - l) ** 2)
                if sigma > 0:
                    effect = np.exp(-grid_distance ** 2 / (2 * sigma ** 2))
                else:
                    if grid_distance == 0:
                        effect = 1
                    else:
                        effect = 0
                direction = x - self.centers[k, l, :]
                p = self.centers[k, l, :]
                self.centers[k, l, :] = self.centers[k, l, :] + eta * effect * direction

    def train(self, data, epochs, eta, sigma_min, sigma_max, gif_name, save_gif=False):
        """
        Trains the SOM on the given data for the given number of epochs
        """
        # resize grid
        for d in range(data.shape[0]):
            d_span = np.max(data[d, :]) - np.min(data[d, :])
            d_min = np.min(data[d, :])
            self.centers[:, :, d] *= d_span
            self.centers[:, :, d] += d_min

        anim = Animator()
        for i in range(epochs):

            anim.save_frame(data, self.centers.copy())
            sigma = sigma_max - (sigma_max - sigma_min) * i / epochs
            for ix, x in enumerate(data.T):

                if i % 10 == 0 and ix == 0 and False:
                    if data.shape[0] == 3:
                        self.plot_grid_3d(data, show=True)
                    if data.shape[0] == 2:
                        self.plot_grid_2d(data, show=True)
                    self.update(x, eta, sigma)

                else:
                    self.update(x, eta, sigma)

        if save_gif:
            anim.save_png_sequence(gif_name)
            anim.save(gif_name + '.gif')

    def plot_grid_3d(self, data=None, show=True):
        """
        Plots the grid of centers with lines connecting them, and optionally the data
        """

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # plot the centers
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                ax.scatter(self.centers[i, j, 0], self.centers[i, j, 1], self.centers[i, j, 2], color='red')

        # plot the lines
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if i < self.shape[0] - 1:
                    ax.plot([self.centers[i, j, 0], self.centers[i + 1, j, 0]],
                            [self.centers[i, j, 1], self.centers[i + 1, j, 1]],
                            [self.centers[i, j, 2], self.centers[i + 1, j, 2]], color='black')
                if j < self.shape[1] - 1:
                    ax.plot([self.centers[i, j, 0], self.centers[i, j + 1, 0]],
                            [self.centers[i, j, 1], self.centers[i, j + 1, 1]],
                            [self.centers[i, j, 2], self.centers[i, j + 1, 2]], color='black')

        # plot the data
        if data is not None:
            ax.scatter(data[0, :], data[1, :], data[2, :])

        # set view to angle of plane
        ax.view_init(30, 30)

        if show:
            plt.show()

    def plot_grid_2d(self, data=None, show=True):
        """
        Plots the grid of centers with lines connecting them, and optionally the data
        """

        # plot the centers
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                plt.scatter(self.centers[i, j, 0], self.centers[i, j, 1], color='red')

        # plot the lines
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if i < self.shape[0] - 1:
                    plt.plot([self.centers[i, j, 0], self.centers[i + 1, j, 0]],
                             [self.centers[i, j, 1], self.centers[i + 1, j, 1]], color='black')
                if j < self.shape[1] - 1:
                    plt.plot([self.centers[i, j, 0], self.centers[i, j + 1, 0]],
                             [self.centers[i, j, 1], self.centers[i, j + 1, 1]], color='black')

        # plot the data
        if data is not None:
            plt.scatter(data[0, :], data[1, :])

        if show:
            plt.show()


if __name__ == '__main__':
    display_data()

    SOM = SOM(shape=(10, 10), dimension=3)

    data = get_surface_dots(n_points=1000)

    SOM.train(data, epochs=300, eta=0.51, sigma_min=0.1, sigma_max=0.1, gif_name='images/3_3/bad_fit_SOM', save_gif=True)

    SOM.plot_grid_3d(data)
