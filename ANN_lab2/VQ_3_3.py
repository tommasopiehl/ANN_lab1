import numpy as np
from matplotlib import pyplot as plt

from ANN_lab2.SOM_3_3 import SOM
from ANN_lab2.lab3_main import RBF_Network, display_model_fit
from sklearn.metrics import mean_squared_error


class VQ_Network:
    def __init__(self,
                 hidden_shape,
                 n_features,
                 output_dim,
                 som_min_sigma,
                 som_max_sigma,
                 rbf_sigma):
        self.som = SOM(hidden_shape, dimension=n_features, random=True)
        self.n_hidden = 1
        self.hidden_shape = hidden_shape
        for i in range(len(hidden_shape)):
            self.n_hidden *= hidden_shape[i]
        self.rbf = RBF_Network(self.n_hidden, output_dim, sigma=rbf_sigma)

        self.som_min_sigma = som_min_sigma
        self.som_max_sigma = som_max_sigma

    def train(self, data, labels, epochs_som, epochs_rbf, eta_som, eta_rbf, gif_name, sigma_min=None, sigma_max=None, ):

        if sigma_min is not None:
            self.som_min_sigma = sigma_min
        if sigma_max is not None:
            self.som_max_sigma = sigma_max

        self.som.train(data.T, epochs_som, eta_som, self.som_min_sigma, self.som_max_sigma, gif_name)

        # get the centers
        self.rbf.centers = self.som.centers.reshape(self.n_hidden, -1)

        # train the rbf
        self.rbf.fit(data, labels, weights='delta_rule', skip_centers=True, epochs=epochs_rbf, learning_rate=eta_rbf)

    def predict(self, data):
        return self.rbf.predict(data)


if __name__ == '__main__':
    # import data from .dat file
    train_data = np.loadtxt("data_lab2/ballist.dat")
    test_data = np.loadtxt("data_lab2/balltest.dat")

    # split data into input and output
    train_input = train_data[:, :2]
    train_output = train_data[:, 2:]

    test_input = test_data[:, :2]
    test_output = test_data[:, 2:]

    # Hyperparameters
    epochs_som = 500
    epochs_rbf = 200
    eta_som = 0.005
    eta_rbf = 0.005
    sigma_min = 0.1
    sigma_max = 2.0
    hidden_shape = (4, 4)
    n_features = 2
    output_dim = 2
    rbf_sigma = 0.1

    n_rbf_nodes = 1
    for i in range(len(hidden_shape)):
        n_rbf_nodes *= hidden_shape[i]

    # create the rbf network
    rbf_net = RBF_Network(
        n_rbf=n_rbf_nodes,
        n_output=n_features,
        sigma=rbf_sigma
    )
    rbf_net.fit(
        train_input,
        train_output,
        centers="equidistant",
        weights='delta_rule',
        epochs=epochs_rbf,
        learning_rate=eta_rbf
    )

    # create the vq network
    vq_net = VQ_Network(
        hidden_shape=hidden_shape,
        n_features=n_features,
        output_dim=output_dim,
        som_min_sigma=sigma_min,
        som_max_sigma=sigma_max,
        rbf_sigma=rbf_sigma
    )

    # train the network
    vq_net.train(
        train_input,
        train_output,
        epochs_som=epochs_som,
        epochs_rbf=epochs_rbf,
        eta_som=eta_som,
        eta_rbf=eta_rbf,
        gif_name="vq_net"
    )

    # predict the training data
    train_prediction_vq = vq_net.predict(train_input)
    mse = mean_squared_error(train_output, train_prediction_vq)
    print("MSE for training data (vector quantization): ", mse)

    train_prediction_rbf = rbf_net.predict(train_input)
    mse = mean_squared_error(train_output, train_prediction_rbf)
    print("MSE for training data (rbf): ", mse)

    plt.cla()
    plt.clf()

    display_model_fit(train_data, train_output, rbf_net, f"RBF Network with 16 centers", save=True)
    display_model_fit(train_data, train_output, vq_net.rbf, f"VQ Network with 16 centers", save=True)

    # predict the test data
    test_prediction_vq = vq_net.predict(test_input)
    mse = mean_squared_error(test_output, test_prediction_vq)
    print("MSE for test data (vector quantization): ", mse)

    test_prediction_rbf = rbf_net.predict(test_input)
    mse = mean_squared_error(test_output, test_prediction_rbf)
    print("MSE for test data (rbf): ", mse)
