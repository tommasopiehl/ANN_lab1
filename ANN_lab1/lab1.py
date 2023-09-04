import random

import numpy as np
import matplotlib.pyplot as plt


def init_w(n_classes, n_features, bias=True):
    """
    This function initializes the weights of the perceptron, with one for each input and
    if desired one for the bias for each class.
    :param n_classes: int
    :param n_features: int
    :param bias: bool
    :return: np.array
    """

    w = np.random.rand(n_classes, n_features + (1 if bias else 0)) - 0.5

    return w


def generate_data(n, mA, mB, sigma, bias=True):
    """
    This function generates data from two classes, A and B, with a normal distribution.
    It will add a bias term if desired.
    :param n: number of data points
    :param mA: the mean of class A
    :param mB: the mean of class B
    :param sigma: the standard deviation of the normal distributions
    :param bias: a boolean indicating whether to add a bias term
    :return:
    """

    # Generate points from the two classes from a normal distribution
    class_a = np.random.multivariate_normal(mA, [[sigma, 0], [0, sigma]], int(n / 2)).T
    class_b = np.random.multivariate_normal(mB, [[sigma, 0], [0, sigma]], int(n / 2)).T

    # Add dummy variable for bias
    if bias:
        class_a = np.vstack((class_a, np.ones(int(n / 2))))
        class_b = np.vstack((class_b, np.ones(int(n / 2))))

    # Create data matrix
    data = np.hstack((class_a, class_b))

    # Create label vector
    labels = np.zeros(n)
    labels[int(n / 2):] = 1  # Set labels for class B to 1

    # Shuffle data and labels
    shuffle = np.random.permutation(n)
    data = data[:, shuffle]
    labels = labels[shuffle]

    return data, labels, class_a, class_b


def display_decision_boundary(w,
                              class_a,
                              class_b,
                              bias=True,
                              highlight_point=None,
                              highlight_color='y',
                              title=None,
                              adjustment=None,
                              show=True,
                              line_color='b'):
    """
    This function displays the decision boundary of the perceptron, as well as the data points.
    It will also show the normal vector of the decision boundary. If desired, it can also show
    a point to highlight and an adjustment vector.
    :param w: weights of the perceptron
    :param class_a: group A
    :param class_b: group B
    :param bias: whether the perceptron has a bias term
    :param highlight_point: the coordinates of a point to highlight
    :param highlight_color: the color of the highlighted point
    :param title: the title of the plot
    :param adjustment: the adjustment vector
    :param show: whether to show the plot
    :param line_color: the color of the decision boundary
    :return:
    """

    # Calculate the slope and intercept of the decision boundary
    k = - (w[0, 0] / w[0, 1])
    if bias:
        m = - w[0, 2] / w[0, 1]
    else:
        m = 0

    # Plot the decision boundary
    x_range = np.linspace(-4, 4, 100)
    y_range = k * x_range + m
    plt.plot(x_range, y_range, c=line_color)

    # Plot the data points
    plt.scatter(class_a[0], class_a[1], c='b')
    plt.scatter(class_b[0], class_b[1], c='r')

    # make window square and set the title
    plt.axis([-4, 4, -4, 4])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)

    # Add normal vector
    plt.quiver(0, m, w[0, 0], w[0, 1], scale=1, scale_units='xy', angles='xy')

    legend = ['Decision boundary',
              'Negative class',
              'Positive class',
              'Normal vector',
              'Point to highlight',
              'Adjustment vector',
              'Axis over origin']

    # Show point to highlight
    if highlight_point is not None:
        plt.scatter(highlight_point[0], highlight_point[1], c=highlight_color, marker='*')
    else:
        legend.remove('Point to highlight')

    # Show adjustment vector
    if adjustment is not None:
        plt.quiver(w[0, 1], w[0, 0] + m, adjustment[0], adjustment[1], scale=1, scale_units='xy', angles='xy',
                   color='g')
    else:
        legend.remove('Adjustment vector')

    # Add axis over the origin as a dotted line with nothcing
    plt.plot([-4, 4], [0, 0], c='k', ls='--')
    plt.plot([0, 0], [-4, 4], c='k', ls='--')

    # Add legend (size 8, upper left corner)
    plt.legend(legend, loc='upper left', fontsize=8)

    # Show the plot if desired
    if show:
        plt.show()


# Initialize hyperparameters

def run_perceptron_experiment(
        n=100,
        lr=0.01,
        n_classes=1,
        n_features=2,
        mA=[2.0, 2.0],
        mB=[-2.0, -2.0],
        sigma=0.5,
        n_epochs=20,
        bias=False,
        draw=1.0):
    data, labels, class_a, class_b = generate_data(n, mA, mB, sigma, bias=bias)
    w = init_w(n_classes, n_features, bias=bias)

    # Plot the decision boundary
    display_decision_boundary(w, class_a, class_b, bias=bias, title="Initial decision boundary")

    # Classify data
    all_good = True
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}")
        all_good = True
        for i in range(n):
            y_pred_i = np.dot(w, data[:, i])  # Calculate the output of the perceptron
            if y_pred_i <= 0:
                y_pred_i = 0
            else:
                y_pred_i = 1
            if y_pred_i != labels[i]:
                if y_pred_i == 0:
                    highlight_color = 'b'
                else:
                    highlight_color = 'r'
                # Only show the first misclassified point per epoch with probability draw
                if all_good:
                    display_decision_boundary(w,
                                              class_a,
                                              class_b,
                                              bias=bias,
                                              highlight_point=data[:, i],
                                              highlight_color=highlight_color,
                                              adjustment=(labels[i] - y_pred_i) * data[:, i],
                                              show=True, title="before update")
                all_good = False
                w = w + lr * (labels[i] - y_pred_i) * data[:, i]  # Update the weights
                if y_pred_i == 0:
                    highlight_color = 'b'
                else:
                    highlight_color = 'r'
                if all_good:
                    display_decision_boundary(w, class_a, class_b, bias=bias, highlight_point=data[:, i],
                                              highlight_color=highlight_color, title="after update")
        if all_good:
            print("All good in epoch ", epoch + 1)
            break
    if not all_good:
        print("Not all good after ", n_epochs, " epochs")

    # Plot the decision boundary
    display_decision_boundary(w, class_a, class_b, bias=bias, title="Final decision boundary")
    print("Final weights: ", w)

    # Classify data and calculate accuracy
    correct = 0
    y_pred = np.zeros(n)
    for i in range(n):
        y_pred_i = np.dot(w, data[:, i])
        if y_pred_i <= 0:
            y_pred_i = 0
        else:
            y_pred_i = 1
        y_pred[i] = y_pred_i
        if y_pred_i == labels[i]:
            correct += 1
    print("Accuracy: ", correct / n)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def shuffle(data, labels):
    # Shuffle the data and labels
    p = np.random.permutation(len(labels))
    return data[:, p], labels[p]


def run_batch_training_experiment(n=100,
                                  lr=0.01,
                                  batch_size=10,
                                  n_classes=1,
                                  n_features=2,
                                  mA=[2.0, 2.0],
                                  mB=[-2.0, -2.0],
                                  sigma=0.5,
                                  n_epochs=20,
                                  bias=True,
                                  draw=1.0):
    data, labels, class_a, class_b = generate_data(n, mA, mB, sigma, bias=bias)
    w = init_w(n_classes, n_features, bias=bias)

    # Plot the decision boundary
    display_decision_boundary(w, class_a, class_b, bias=bias, title="Initial decision boundary")
    input("Press Enter to continue...")
    error_history = []
    epoch_error_history = []
    epoch_error = 0
    prev_epoch_error = float('inf')
    # Train the perceptron
    for epoch in range(n_epochs):

        # Shuffle data
        data, labels = shuffle(data, labels)

        for batch in range(int(n / batch_size)):
            batch_data = data[:, batch * batch_size:(batch + 1) * batch_size]
            batch_labels = labels[batch * batch_size:(batch + 1) * batch_size]
            y_pred = sigmoid(np.dot(w, batch_data))

            e = batch_labels - y_pred
            print(f"error: {np.sum(e**2)}")
            error_history.append(np.sum(e**2))
            epoch_error += np.sum(e**2)
            w = w + lr * np.dot(e, batch_data.T)

        epoch_error = np.sum((sigmoid(np.dot(w, data)) - labels)**2)
        epoch_error_history.append(epoch_error/n)
        if abs(epoch_error - prev_epoch_error)/n < 0.001:
            print(f"Epoch {epoch + 1} converged")
            break
        prev_epoch_error = epoch_error
        if random.random() < draw:
            display_decision_boundary(w, class_a, class_b, bias=bias, title="updated decision boundary")


    # Plot the decision boundary
    display_decision_boundary(w, class_a, class_b, bias=bias, title="Final decision boundary")
    epoch_error = np.sum((sigmoid(np.dot(w, data)) - labels) ** 2)
    epoch_error_history.append(epoch_error / n)
    # Display error history with markers per batch and a line per epoch
    plt.plot(error_history, marker='o', markersize=1, linewidth=0.8)

    #lines and points per epoch
    x_batch = np.arange(0, len(epoch_error_history) * (n / batch_size), n / batch_size)
    plt.plot(x_batch, epoch_error_history, marker='o', markersize=2.5, linewidth=2)
    for i in range(1, n_epochs):
        plt.axvline(i * (n / batch_size), color='r', linestyle='--', linewidth=0.5)

    plt.title("Errors per batch")
    plt.show()


# input("Run experiment 1: Linearly seperable data split over the origin, without bias term")
# run_experiment(n=100,
#                lr=0.01,
#                n_classes=1,
#                n_features=2,
#                mA=[2.0, 2.0],
#                mB=[-2.0, -2.0],
#                sigma=0.5,
#                n_epochs=10,
#                bias=False)

# input("Run experiment 2: Linearly seperable data, not split over the origin, without bias term")
# run_experiment(n=100,
#                lr=0.01,
#                n_classes=1,
#                n_features=2,
#                mA=[3.0, 3.0],
#                mB=[1.0, 1.0],
#                sigma=0.3,
#                n_epochs=10,
#                bias=False,
#                draw=0.1)

# input("Run experiment 3: Linearly seperable data, not split over the origin, with bias term")
# run_perceptron_experiment(n=100,
#                           lr=0.1,
#                           n_classes=1,
#                           n_features=2,
#                           mA=[3.0, 3.0],
#                           mB=[1.0, 1.0],
#                           sigma=0.3,
#                           n_epochs=10,
#                           bias=True,
#                           draw=1.0)

run_batch_training_experiment(n=100,
                              lr=0.01,
                              batch_size=1,
                              n_classes=1,
                              n_features=2,
                              mA=[2.0, 1.0],
                              mB=[2.0, -1.0],
                              sigma=0.5,
                              n_epochs=20,
                              bias=True,
                              draw=0.1)
