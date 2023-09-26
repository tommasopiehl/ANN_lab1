import numpy as np
import scipy.interpolate as interpolate
import scipy.signal as signal
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from ANN_lab3.animator_3_1 import Animator_3_1
from ANN_lab3.lab3_main import RBF_Network

def dual_plot(x, y1, y2, title, xlabel, ylabel, legend1, legend2):
    """
    Plots two curves in the same plot
    """
    fig, ax = plt.subplots()
    ax.plot(x, y1, label=legend1)
    ax.plot(x, y2, label=legend2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.show()

# Generate data

# Span
x_min = 0
x_max = 2*np.pi
step_train = 0.1
step_test = 0.05

# Generate training data

x_train = np.linspace(x_min, x_max, int((x_max - x_min) / step_train))
y_1_train = np.sin(2*x_train)
y_2_train = signal.square(2*x_train)

x_train = x_train.reshape(-1, 1)
y_1_train = y_1_train.reshape(-1, 1)
y_2_train = y_2_train.reshape(-1, 1)

# Generate test data

x_test = np.linspace(x_min, x_max, int((x_max - x_min) / step_test))
y_1_test = np.sin(2*x_test)
y_2_test = signal.square(2*x_test)

x_test = x_test.reshape(-1, 1)
y_1_test = y_1_test.reshape(-1, 1)
y_2_test = y_2_test.reshape(-1, 1)


# Test RBF networks

# parameters
n_centers = 62
sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]
epochs = 100
learning_rate = 0.1
weight_method = 'least_squares'
center_method = 'kmeans'

# RBF network for sin.png(2x)

sig_mse_list = []

for sig in sigmas:
    anim = Animator_3_1()
    mse_list = []
    found_01 = False
    found_001 = False
    found_0001 = False
    for cent in range(1, n_centers + 1):
        model = RBF_Network(cent, 1, sig)
        model.fit(x_train, y_1_train, centers=center_method, weights=weight_method, epochs=epochs, learning_rate=learning_rate)
        y_test_pred = model.predict(x_test)
        # MSE
        mse = mean_squared_error(y_1_test, y_test_pred)
        mae = mean_absolute_error(y_1_test, y_test_pred)
        if mae < 0.1 and not found_01:
            print(f"Sin(2x) - {cent} centers, sigma = {sig}, MSE = {mse} MAE = {mae}")
            found_01 = True
        if mae < 0.01 and not found_001:
            print(f"Sin(2x) - {cent} centers, sigma = {sig}, MSE = {mse} MAE = {mae}")
            found_001 = True
        if mae < 0.001 and not found_0001:
            print(f"Sin(2x) - {cent} centers, sigma = {sig}, MSE = {mse} MAE = {mae}")
            found_0001 = True
            break
        #print(f"Sin(2x) - {cent} centers, sigma = {sig}, MSE = {mse}")
        mse_list.append(mse)

        # Plot
        #dual_plot(x_test, y_1_test, y_test_pred, f"Sin(2x) - {cent} centers, sigma = {sig}", "x", "y", "Real", "Predicted")
        anim.save_frame(x_test, y_1_test, y_test_pred, f"Sin(2x) - {cent} centers, sigma = {sig}", "x", "y", "Real", "Predicted")
    #anim.save_png_sequence(f"Sin(2x)-sigma={sig}")
    #anim.save(f"Sin(2x)-sigma={sig}")

    # Plot MSE
    #sig_mse_list.append(mse_list)

# plt.clf()
# plt.cla()
# for i, sig in enumerate(sigmas):
#     plt.plot(range(1, n_centers + 1), sig_mse_list[i], label=f"sigma = {sig}")
# plt.title("MSE for Sin(2x)")
# plt.xlabel("Number of centers")
# plt.ylabel("MSE")
# plt.legend()
# plt.show()


# RBF network for square.png(2x)

sig_mse_list = []

for sig in sigmas:
    mse_list = []
    found_01 = False
    found_001 = False
    found_0001 = False
    anim = Animator_3_1()
    for cent in range(1, n_centers + 1):
        model = RBF_Network(cent, 1, sig)
        model.fit(x_train, y_2_train, centers=center_method, weights=weight_method, epochs=epochs, learning_rate=learning_rate)
        y_test_pred = model.predict(x_test)
        # MSE
        mse = mean_squared_error(y_2_test, y_test_pred)
        mae = mean_absolute_error(y_2_test, y_test_pred)

        if mae < 0.1 and not found_01:
            print(f"Square(2x) - {cent} centers, sigma = {sig}, MSE = {mse} MAE = {mae}")
            found_01 = True
        if mae < 0.01 and not found_001:
            print(f"Square(2x) - {cent} centers, sigma = {sig}, MSE = {mse} MAE = {mae}")
            found_001 = True
        if mae < 0.001 and not found_0001:
            print(f"Square(2x) - {cent} centers, sigma = {sig}, MSE = {mse} MAE = {mae}")
            found_0001 = True
            break
        #print(f"Square(2x) - {cent} centers, sigma = {sig}, MSE = {mse}")
        mse_list.append(mse)

        # Plot
        #dual_plot(x_test, y_2_test, y_test_pred, f"Square(2x) - {cent} centers, sigma = {sig}", "x", "y", "Real", "Predicted")
        anim.save_frame(x_test, y_2_test, y_test_pred, f"Square(2x) - {cent} centers, sigma = {sig}", "x", "y", "Real", "Predicted")
    #anim.save_png_sequence(f"Square(2x)-sigma={sig}")
    #anim.save(f"Square(2x)-sigma={sig}")

    # Plot MSE
    sig_mse_list.append(mse_list)

plt.clf()
plt.cla()
for i, sig in enumerate(sigmas):
    plt.plot(range(1, n_centers + 1), sig_mse_list[i], label=f"sigma = {sig}")
plt.title("MSE for Square(2x)")
plt.xlabel("Number of centers")
plt.ylabel("MSE")
plt.legend()
plt.show()

