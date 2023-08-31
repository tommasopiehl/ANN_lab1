import numpy as np
import matplotlib.pyplot as plt


def init_w(n_classes, n_features):
    """
    This function initializes the weights of the perceptron, with one for each input and one for the bias for each class.
    :param n_classes: int
    :param n_features: int
    :return: np.array
    """

    w = np.random.rand(n_classes, n_features + 1) - 0.5

    return w


def adjust_w(w_old, delta_w):
    w_new = w_old + delta_w

    return w_new


def plot_bound(w, X):
    wT = w.transpose()
    x_range = np.linspace(X[0].min(), X[0].max(), 100)
    y_range = (-wT[0] * x_range - wT[2]) / wT[1]

    return x_range, y_range

# Initialize variables
n = 10
lr = 1
n_classes = 1
n_features = 2
class_a = np.zeros([2, int(n / 2)])
class_b = np.zeros([2, int(n / 2)])
data = np.ones([3, n])
y_classification = np.zeros(n)
mA = [1.0, 1.0]
mB = [-1.0, -1.0]
sigma = 0.5

# Generate data
# classA[0][:] = np.random.rand(int(n / 2)) * sigma + mA[0]
# classA[1][:] = np.random.rand(int(n / 2)) * sigma + mA[1]
# classB[0][:] = np.random.rand(int(n / 2)) * sigma + mB[0]
# classB[1][:] = np.random.rand(int(n / 2)) * sigma + mB[1]

class_a = np.random.multivariate_normal(mA, [[sigma, 0], [0, sigma]], int(n / 2)).T
class_b = np.random.multivariate_normal(mB, [[sigma, 0], [0, sigma]], int(n / 2)).T

# Add dummy variable for bias
class_a = np.vstack((class_a, np.ones(int(n / 2))))
class_b = np.vstack((class_b, np.ones(int(n / 2))))

#Display class B and A in scatter plot as crosses and circles respectively
plt.scatter(class_a[0], class_a[1], c='r')
plt.scatter(class_b[0], class_b[1], c='b')
plt.show()
input("Press Enter to continue...")



# Initialize weights
w = init_w(n_classes, n_features)
dw = np.zeros([n, 3])

# Display decision boundary
def display_decision_boundary(w, class_a, class_b):
    k = w[0,0]/w[0,1]
    m = w[0,2]

    x_range = np.linspace(-2, 2, 100)
    y_range = k*x_range + m
    plt.plot(x_range, y_range)
    plt.scatter(class_a[0], class_a[1], c='r')
    plt.scatter(class_b[0], class_b[1], c='b')
    plt.show()

display_decision_boundary(w, class_a, class_b)

# Classify data

for i in range(5):
    y_classification = []
    y_value = []
    for i, x_i in enumerate(class_a.T):
        y_i = np.dot(w, x_i)
        y_value.append(y_i)
        y_i = int(y_i > 0)
        y_classification.append(y_i)
        if y_i == 0:
            delta_w = lr * x_i
        else:
            delta_w = 0
        w += delta_w
        display_decision_boundary(w, class_a, class_b)

        x_i = class_b.T[i]
        y_i = np.dot(w, x_i)
        y_value.append(y_i)
        y_i = int(y_i > 0)
        y_classification.append(y_i)
        if y_i == 1:
            delta_w = lr * x_i
        else:
            delta_w = 0
        w += delta_w
        display_decision_boundary(w, class_a, class_b)



print(y_classification)
input("Press Enter to continue...")




# y_i = np.dot(w, data[:3]) # Calculate the output of the perceptron
# y_i = np.sum(y_i, axis=0)
# y = np.zeros(n)
# check = 0
# labels = np.zeros(n)
# error_ls = []


# PERCEPTRON LEARNING

for k in range(1000):

    error = 0

    for x_i in range(len(y_i)):
        if y_i[x_i] <= 0:
            y_classification[x_i] = 0
            plt.scatter(data[0][x_i], data[1][x_i], c='r')
        else:
            y_classification[x_i] = 1
            plt.scatter(data[0][x_i], data[1][x_i], c='b')
    plt.show()

    for j in range(len(y_classification)):
        if y_classification[j] > labels[j]:
            dw[j][:] = dw[j][:] - lr * data_T[j][:]
            error += 1
        elif y_classification[j] < labels[j]:
            dw[j][:] = dw[j][:] + lr * data_T[j][:]
            error += 1

    w = adjust_w(w, dw)
    y_i = np.matmul(w, data[:3])
    y_i = np.sum(y_i, axis=0)
    error_ls.append(error / n)

    if error == 0:
        break

    # if check % 100 == 0:
    #     # x, y = plot_bound(w, data)
    #     # plt.plot(x,y)
    #     plt.show()

# DELTA RULE

w_D = init_w(n)
dw_D = np.zeros([n, 3])

y_i_D = np.dot(w_D, data[:3])
y_i_D = np.sum(y_i_D, axis=0)
y_D = np.zeros(n)
check_D = 0
error_ls_D = []

error = 0

for j in range(len(y_D)):
    if y_i_D[x_i] > labels[j]:
        dw_D[j][:] = dw_D[j][:] - lr * data_T[j][:]
        error += 1
    elif y_i_D[x_i] < labels[j]:
        dw_D[j][:] = dw_D[j][:] + lr * data_T[j][:]
        error += 1

for x_i in range(len(y_i_D)):
    if y_i_D[x_i] <= 0:
        y_D[x_i] = 0
        plt.scatter(data[0][x_i], data[1][x_i], c='r')
    else:
        y_D[x_i] = 1
        plt.scatter(data[0][x_i], data[1][x_i], c='b')

w_D = adjust_w(w_D, dw_D)
y_i_D = np.matmul(w_D, data[:3])
y_i_D = np.sum(y_i_D, axis=0)
error_ls_D.append(error / n)

# if error == 0:
#     break

# plt.plot(error_ls, c='r')
plt.plot(error_ls_D, c='b')
plt.show()
