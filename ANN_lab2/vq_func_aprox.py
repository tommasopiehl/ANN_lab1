import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ANN_lab2.VQ_3_3 import VQ_Network
from ANN_lab2.animator_3_1 import Animator_3_1
from ANN_lab2.experiment_3_1 import dual_plot

if __name__ == '__main__':

    # Hyperparameters
    epochs_som = 100
    epochs_rbf = 100
    eta_som = 0.01
    eta_rbf = 0.01
    sigma_min = 0.1
    sigma_max = 2.0
    hidden_shape = (4, 4)
    n_features = 1
    output_dim = 1
    rbf_sigma = 0.1

    n_centers = 30
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Data generation

    # Span
    x_min = 0
    x_max = 2 * np.pi
    step_train = 0.1
    step_test = 0.05

    # Generate training data

    x_train = np.linspace(x_min, x_max, int((x_max - x_min) / step_train))
    noise = np.random.normal(0, 0.1, x_train.shape)
    y_1_train = np.sin(2 * x_train) + noise
    y_2_train = signal.square(2 * x_train) + noise

    x_train = x_train.reshape(-1, 1)
    noise = np.random.normal(0, 0.1, x_train.shape)
    y_1_train = y_1_train.reshape(-1, 1) + noise
    y_2_train = y_2_train.reshape(-1, 1) + noise

    # Generate test data

    x_test = np.linspace(x_min, x_max, int((x_max - x_min) / step_test))
    noise = np.random.normal(0, 0.1, x_test.shape)
    y_1_test = np.sin(2 * x_test + (noise * x_test))
    y_2_test = signal.square(2 * x_test + (noise * x_test))

    x_test = x_test.reshape(-1, 1)
    y_1_test = y_1_test.reshape(-1, 1)
    y_2_test = y_2_test.reshape(-1, 1)

    # testing

    sig_mse_list = []

    for sig in sigmas:
        anim = Animator_3_1()
        mse_list = []
        for cent in range(1, n_centers + 1):
            print(f"sig: {sig}, cent: {cent}")
            hidden_shape = (cent, 1)
            model = VQ_Network(
                hidden_shape=hidden_shape,
                n_features=n_features,
                output_dim=output_dim,
                som_min_sigma=sigma_min,
                som_max_sigma=sigma_max,
                rbf_sigma=rbf_sigma
            )

            # train the network
            model.train(
                x_train,
                y_1_train,
                epochs_som=epochs_som,
                epochs_rbf=epochs_rbf,
                eta_som=eta_som,
                eta_rbf=eta_rbf,
                gif_name="vq_net"
            )
            y_test_pred = model.predict(x_test)
            # MSE

            mse = mean_squared_error(y_1_test, y_test_pred)
            mae = mean_absolute_error(y_1_test, y_test_pred)

            # print(f"Sin(2x) - {cent} centers, sigma = {sig}, MSE = {mse}")
            mse_list.append(mse)

            # Plot
            #dual_plot(x_test, y_1_test, y_test_pred, f"Sin(2x) - {cent} centers, sigma = {sig}", "x", "y", "Real",
            #          "Predicted")
            anim.save_frame(x_test, y_1_test, y_test_pred, f"Sin(2x) - {cent} centers, sigma = {sig}", "x", "y", "Real",
                            "Predicted")

        anim.save_png_sequence(f"Sin(2x)-sigma={sig}")
        anim.save(f"Sin(2x)-sigma={sig}")

    plt.clf()
    plt.cla()
    for i, sig in enumerate(sigmas):
        try:
            plt.plot(range(1, n_centers), sig_mse_list[i], label=f"sigma = {sig}")
        except:
            pass
    plt.title("MSE for Sin(2x)")
    plt.xlabel("Number of centers")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


    sig_mse_list = []
