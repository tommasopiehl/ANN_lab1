import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import sklearn.mixture as mix


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

class RBF_Network:
    def __init__(self, n_rbf, n_output, sigma=0.5, sigmas=None):
        self.n_rbf = n_rbf
        self.n_output = n_output
        if sigmas is None:
            self.sigmas = np.full(n_rbf, sigma)
        else:
            self.sigmas = sigmas
        self.centers = None
        self.weights = None

    def __get_random_centers__(self, data):
        self.centers = data[np.random.choice(len(data), self.n_rbf, replace=False)]

    def __get_kmeans_centers__(self, data):
        self.centers = data[np.random.choice(len(data), self.n_rbf, replace=False)]
        for i in range(100):
            clusters = [[] for _ in range(self.n_rbf)]
            for j in range(len(data)):
                distances = np.zeros(self.n_rbf)
                for k in range(self.n_rbf):
                    distances[k] = np.sqrt(np.sum(np.square(data[j] - self.centers[k])))
                clusters[np.argmin(distances)].append(data[j])
            for j in range(self.n_rbf):
                self.centers[j] = np.mean(clusters[j], axis=0)

    def __get_gaussian_mixture_centers__(self, data):
        gmm = mix.GaussianMixture(n_components=self.n_rbf, covariance_type='diag')
        gmm.fit(data)
        self.centers = gmm.means_
        self.sigmas = np.sqrt(gmm.covariances_)[:, 0]

    def fit(self, data, classification, centers='random'):
        if centers == 'random':
            self.__get_random_centers__(data)
        elif centers == 'kmeans':
            self.__get_kmeans_centers__(data)
        elif centers == 'gaussian_mixture':
            self.__get_gaussian_mixture_centers__(data)
        phi = np.zeros((len(data), self.n_rbf))
        for i in range(len(data)):
            for j in range(self.n_rbf):
                phi[i, j] = rbf(data[i], self.centers[j], self.sigmas[j])
        self.weights = np.linalg.pinv(phi) @ classification

    def predict(self, data):
        phi = np.zeros((len(data), self.n_rbf))
        for i in range(len(data)):
            for j in range(self.n_rbf):
                phi[i, j] = rbf(data[i], self.centers[j], self.sigmas[j])
        return phi @ self.weights

    def score(self, data, classification):
        return np.sum(np.square(self.predict(data) - classification)) / len(data)


def display_model_fit(data, labels, model, title, save=False):
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.scatter(model.centers[:, 0], model.centers[:, 1], c='red')

    for i in range(len(model.centers)):
        circle = plt.Circle(model.centers[i], model.sigmas[i], color='red', fill=False)
        plt.gcf().gca().add_artist(circle)

    plt.title(title)
    plt.axis('equal')

    if save:
        plt.savefig(f"{title}.png")

    plt.show()

if __name__ == '__main__':

    points = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
    n_points = [100, 100, 100, 100]
    sigmas = [0.5, 0.5, 0.5, 0.5]
    classes = [1, -1, 1, -1]

    for fit in ['random', 'kmeans', 'gaussian_mixture']:

        data, labels = generate_data(points, n_points, sigmas, classes)

        model = RBF_Network(4, 1, 0.5)
        model.fit(data, labels, centers=fit)

        print(f"Score for {fit} centers: {model.score(data, labels)}")

        # display the centres and variance of the RBFs
        display_model_fit(data, labels, model, f"RBF Network with {fit} centers", save=True)



