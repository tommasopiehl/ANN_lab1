import numpy as np
import matplotlib.pyplot as plt


def init_w(n_classes, n_features, bias=True):
    """
    This function initializes the weights of the perceptron, with one for each input and
    one for the bias for each class.
    :param n_classes: int
    :param n_features: int
    :return: np.array
    """

    w = np.random.rand(n_classes, n_features + (1 if bias else 0)) - 0.5

    return w


# Initialize variables
n = 10
lr = 0.5
n_classes = 1
n_features = 2
mA = [1.0, 1.0]
mB = [-1.0, -1.0]
sigma = 0.5

# Generate data

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
w = init_w(n_classes, n_features, bias=True)

# Display decision boundary
def display_decision_boundary(w, class_a, class_b, bias=True, highlight_point=None, highlight_color='r', title = None):

    k = - (w[0,1]/w[0,0])
    if bias:
        m = w[0,2]
    else:
        m = 0

    # Plot the decision boundary
    x_range = np.linspace(-4, 4, 100)
    y_range = k*x_range + m
    plt.plot(x_range, y_range)
    plt.scatter(class_a[0], class_a[1], c='r')
    plt.scatter(class_b[0], class_b[1], c='b')

    # make axis [-3, 3, -3, 3] and window square
    plt.axis([-3, 3, -3, 3])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)

    # Add normal vector
    plt.quiver(0, m, w[0,1], w[0,0], scale=1, scale_units='xy', angles='xy')


    # Highlight point
    if highlight_point is not None:
        plt.scatter(highlight_point[0], highlight_point[1], c=highlight_color, marker='*')
    plt.show()

display_decision_boundary(w, class_a, class_b)

# Create data matrix
data = np.hstack((class_a, class_b))

# Create label vector
labels = np.zeros(n)
labels[int(n / 2):] = 1

# Shuffle data and labels
shuffle = np.random.permutation(n)
data = data[:, shuffle]
labels = labels[shuffle]


# Classify data
for epoch in range(10):
    all_good = True
    for i in range(n):
        y_pred_i = np.dot(w, data[:, i]) # Calculate the output of the perceptron
        if y_pred_i <= 0:
            y_pred_i = 0
        else:
            y_pred_i = 1
        if y_pred_i != labels[i]:
            if y_pred_i == 0:
                highlight_color = 'b'
            else:
                highlight_color = 'r'
            display_decision_boundary(w, class_a, class_b, bias=False, highlight_point=data[:, i], highlight_color=highlight_color, title="before update")
            all_good = False
            w = w - lr * (labels[i] - y_pred_i) * data[:, i] # Update the weights
            if y_pred_i == 0:
                highlight_color = 'r'
            else:
                highlight_color = 'b'
            display_decision_boundary(w, class_a, class_b, bias=False, highlight_point=data[:, i], highlight_color=highlight_color, title="after update")
    if all_good:
        print("All good in epoch ", epoch)
        break

# Plot the decision boundary
display_decision_boundary(w, class_a, class_b, bias=False)



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
