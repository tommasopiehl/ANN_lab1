import numpy as np
import matplotlib.pyplot as plt

def init_w():

    w = np.random.rand(1, 3) - 0.5

    return w

def adjust_w(w_old, delta_w):

    w_new = w_old + delta_w

    return w_new

def plot_bound(w, X):

    x_range = np.linspace(-2, 2, n*100)
    y_range = -(w[0][0]/w[0][1])*x_range - w[0][2]/(w[0][0]**2+w[0][1]**2)**0.5

    return x_range, y_range

n = 100
lr = 0.0001
classA = np.zeros([2,int(n/2)])
classB = np.zeros([2,int(n/2)])
data = np.ones([3, n])
labels = np.zeros(n)
y = np.zeros(n)
mA = [ 1.0, 0.3]
mB = [-1.0, 0.4]
sigma = 0.5

classA[0] = np.random.rand(int(n/2)) * sigma + mA[0]
classA[1] = np.random.rand(int(n/2)) * sigma + mA[1]
classB[0] = np.random.rand(int(n/2)) * sigma + mB[0]
classB[1] = np.random.rand(int(n/2)) * sigma + mB[1]


data[0]= np.concatenate((classA[0], classB[0]))
data[1]= np.concatenate((classA[1], classB[1]))

np.apply_along_axis(np.random.shuffle, axis=1, arr=data)

for i, sample in enumerate(data[0]):
    if sample in classA[0]:
        labels[i] = 1
    elif sample in classB[0]:
        labels[i] = 0

data_T = data.transpose()

w = init_w()
dw = np.zeros([1, 3])
y_i = np.zeros(n)

for i in range(n):
    y_i[i] = np.sum(w[0]*data_T[i])

y = np.zeros(n)
check = 0
error_ls = []
bound = []

#PERCEPTRON LEARNING, borde vara klar

for k in range(100):

    error = 0

    for i in range(len(y_i)):
        if y_i[i] < 0:
            y[i] = 0
            plt.scatter(data[0][i], data[1][i], c='r')
        elif y_i[i] > 0:
            y[i] = 1
            plt.scatter(data[0][i], data[1][i], c='b')

    for j in range(len(y)):
        if y[j] > labels[j]:
            dw = dw - lr*data_T[j]
            error += 1
        elif y[j] < labels[j]:
            dw = dw + lr*data_T[j]
            error += 1

    x_plt, y_plt = plot_bound(w, data)
    plt.plot(x_plt, y_plt)
    plt.xlim([-2, 2])
    plt.ylim([data[1].min(), data[1].max()])
    plt.show()

    w = adjust_w(w, dw)

    for i in range(n):
        y_i[i] = np.sum(w*data_T[i])

    error_ls.append(error/n)

    if error == 0:
        break

plt.plot(error_ls)
plt.show()

#DELTA RULE, inte klar

w_D = init_w()
dw_D = np.zeros([1, 3])
y_i_D = np.zeros(n)

for i in range(n):
    y_i_D[i] = np.sum(w[0]*data_T[i])

y_D = np.zeros(n)
error_ls_D = []

for k in range(100):

    error = 0

    for j in range(len(y_D)):
        if y_i_D[i] > labels[j]:
            dw_D[j] = dw_D[j] - (labels[j]-y_i_D[i])*lr
            error += 1
        elif y_i_D[i] < labels[j]:
            dw_D[j][:] = dw_D[j][:] + (labels[j]-y_i_D[i])*lr
            error += 1

    for i in range(len(y_i_D)):
        if y_i_D[i] <= 0:
            y_D[i] = 0
            #plt.scatter(data[0][i], data[1][i], c='r')
        else:
            y_D[i] = 1
            #plt.scatter(data[0][i], data[1][i], c='b')

    w_D = adjust_w(w_D, dw_D)
    y_i_D = np.dot(w_D, data[:3])
    y_i_D = np.sum(y_i_D, axis=0)
    error_ls_D.append(error/n) 

    if error == 0:
        break

    #plt.show()

#print(error_ls_D)
# plt.plot(error_ls, c='r')
# plt.plot(error_ls_D, c='b')
# plt.show()
