import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import sklearn.mixture as mix
import sklearn.cluster as cluster
from sklearn.metrics import accuracy_score


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
    return np.concatenate(data), np.concatenate(classification).reshape(-1, 1)

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
        kmeans = cluster.KMeans(n_clusters=self.n_rbf, n_init='auto')
        kmeans.fit(data)
        self.centers = kmeans.cluster_centers_

    def __get_gaussian_mixture_centers__(self, data):
        gmm = mix.GaussianMixture(n_components=self.n_rbf, covariance_type='diag')
        gmm.fit(data)
        self.centers = gmm.means_
        self.sigmas = np.sqrt(gmm.covariances_)[:, 0]

    def __pseudo_inverse_fit__(self, phi, classification):
        self.weights = np.linalg.pinv(phi) @ classification

    def __least_squares_fit__(self, phi, classification):
        self.weights = np.linalg.lstsq(phi, classification, rcond=None)[0]

    def __delta_rule_fit__(self, phi, classification, epochs, learning_rate, n_samples):
        self.weights = np.random.rand(self.n_rbf, self.n_output)
        for epoch in range(epochs):
            for i in range(n_samples):

                class_error = classification[i] - np.dot(phi[i, :], self.weights)
                for j in range(self.n_rbf):
                    self.weights[j, :] += learning_rate * class_error * phi[i, j]

                #self.weights += learning_rate * (classification[i] - np.dot(phi[i, :], self.weights)) * phi[i, :]
                #self.weights += learning_rate * (classification[i] - phi[i, :] @ self.weights) * phi[i, :]

    def phi_activation(self, data):
        phi = np.zeros((len(data), self.n_rbf))
        for i in range(len(data)):
            for j in range(self.n_rbf):
                phi[i, j] = rbf(data[i], self.centers[j], self.sigmas[j])
        return phi

    def fit(self,
            data,
            classification,
            centers='random',
            weights='pseudoinverse',
            epochs=100,
            learning_rate=0.1,
            skip_centers=False):

        if not skip_centers:
            if centers == 'random':
                self.__get_random_centers__(data)
            elif centers == 'equidistant':
                if data.shape[1] == 1:
                    self.centers = np.linspace(data.min(), data.max(), self.n_rbf)
                elif data.shape[1] == 2:
                    if np.sqrt(self.n_rbf) % 1 != 0:
                        raise ValueError('Invalid number of RBFs for equidistant centers')

                    n = int(np.sqrt(self.n_rbf))
                    # To make the centers be in the data, not at the edges
                    dist_x = data[:, 0].max() - data[:, 0].min()
                    dist_y = data[:, 1].max() - data[:, 1].min()

                    d_dist_x = 0.5 * dist_x / (n + 1)
                    d_dist_y = 0.5 * dist_y / (n + 1)

                    x = np.linspace(data[:, 0].min() + d_dist_x, data[:, 0].max() - d_dist_x, n)
                    y = np.linspace(data[:, 1].min() + d_dist_y, data[:, 1].max() - d_dist_y, n)
                    self.centers = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
                else:
                    raise ValueError('Invalid number of dimensions for equidistant centers')
            elif centers == 'kmeans':
                self.__get_kmeans_centers__(data)
            elif centers == 'gaussian_mixture':
                self.__get_gaussian_mixture_centers__(data)
            else:
                raise ValueError('Invalid center type: ' + centers)

        phi = self.phi_activation(data)

        if weights == 'pseudoinverse':
            self.__pseudo_inverse_fit__(phi, classification)
        elif weights == 'least_squares':
            self.__least_squares_fit__(phi, classification)
        elif weights == 'delta_rule':
            self.__delta_rule_fit__(phi, classification, epochs, learning_rate, len(data))

    def predict(self, data):
        phi = np.zeros((len(data), self.n_rbf))
        for i in range(len(data)):
            for j in range(self.n_rbf):
                phi[i, j] = rbf(data[i], self.centers[j], self.sigmas[j])
        return phi @ self.weights

    def score(self, data, classification):
        return np.sum(np.square(self.predict(data) - classification)) / len(data)


def display_model_fit(data, labels, model, title, save=False):
    if labels.shape[1] == 2:
        col = [0] * labels.shape[0]
        # rescale to 0-1
        labels = (labels - np.min(labels)) / (np.max(labels) - np.min(labels))

        # set rbg in range 0-1
        for i in range(len(labels)):
            col[i] = (labels[i, 0], labels[i, 1], 0.5)
        plt.scatter(data[:, 0], data[:, 1], c=col)
    else:
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
    sigmas = [0.2, 0.7, 0.1, 0.3]
    classes = [1, -1, 1, -1]

    for fit in ['equidistant', 'random', 'kmeans', 'gaussian_mixture']:

        data, labels = generate_data(points, n_points, sigmas, classes)

        model = RBF_Network(4, 1, 0.5)
        #model.fit(data, labels, centers=fit, weights='least_squares')
        #model.fit(data, labels, centers=fit, weights='pseudoinverse')
        model.fit(data, labels, centers=fit, weights='delta_rule', epochs=100, learning_rate=0.1)

        print(f"Score for {fit} centers: {model.score(data, labels)}")

        # accuracy
        print(f"Accuracy for {fit} centers: {accuracy_score(labels, np.sign(model.predict(data)))}")

        # display the centres and variance of the RBFs
        display_model_fit(data, labels, model, f"RBF Network with {fit} centers", save=True)



