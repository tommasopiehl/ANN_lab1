import numpy as np
import matplotlib.pyplot as plt

def init_w():

    w = np.zeros([3, n])
    w[0][:] = np.random.normal(0, 1, n)
    w[1][:] = np.random.normal(0, 1, n)
    w[2][:] = np.random.normal(0, 1, n)

    w = w.transpose()

    return w

def adjust_w(w_old, delta_w):

    w_new = w_old + delta_w

    return w_new

def plot_bound(w, X):

    wT = w.transpose()
    x_range = np.linspace(X[0].min(), X[0].max(), 100)
    y_range = (-wT[0] * x_range - wT[2]) / wT[1]

    return x_range, y_range

n = 100
lr = 0.001
classA = np.zeros([2,int(n/2)])
classB = np.zeros([2,int(n/2)])
data = np.ones([3, n])
y = np.zeros(n)
mA = [ 1.0, 0.3]
mB = [-1.0, 0.4]
sigma = 0.5

classA[0][:] = np.random.rand(int(n/2)) * sigma + mA[0]
classA[1][:] = np.random.rand(int(n/2)) * sigma + mA[1]
classB[0][:] = np.random.rand(int(n/2)) * sigma + mB[0]
classB[1][:] = np.random.rand(int(n/2)) * sigma + mB[1]

data[0]= np.concatenate((classA[0], classB[0]))
data[1]= np.concatenate((classA[1], classB[1]))

np.apply_along_axis(np.random.shuffle, axis=1, arr=data)

data_T = data.transpose()

w = init_w()
dw = np.zeros([n, 3])

y_i = np.dot(w, data[:3])
y_i = np.sum(y_i, axis=0)
y = np.zeros(n)
check = 0
labels = np.zeros(n)
error_ls = []

for i in range(n):
    if data[0][i] > 0:
        labels[i] = 1

#PERCEPTRON LEARNING

for k in range(1000):

    error = 0

    for i in range(len(y_i)):
        if y_i[i] <= 0:
            y[i] = 0
            #plt.scatter(data[0][i], data[1][i], c='r')
        else:
            y[i] = 1
            #plt.scatter(data[0][i], data[1][i], c='b')

    for j in range(len(y)):
        if y[j] > labels[j]:
            dw[j][:] = dw[j][:] -lr*data_T[j][:]
            error += 1
        elif y[j] < labels[j]:
            dw[j][:] = dw[j][:] + lr*data_T[j][:]
            error += 1
    
    w = adjust_w(w, dw)
    y_i = np.matmul(w, data[:3])
    y_i = np.sum(y_i, axis=0)
    error_ls.append(error/n)

    if error == 0:
        break

    # if check % 100 == 0:
    #     # x, y = plot_bound(w, data)
    #     # plt.plot(x,y)
    #     plt.show()

#DELTA RULE

w_D = init_w()
dw_D = np.zeros([n, 3])

y_i_D = np.dot(w_D, data[:3])
y_i_D = np.sum(y_i_D, axis=0)
y_D = np.zeros(n)
check_D = 0
error_ls_D = []


error = 0

for j in range(len(y_D)):
    if y_i_D[i] > labels[j]:
        dw_D[j][:] = dw_D[j][:] - lr*data_T[j][:]
        error += 1
    elif y_i_D[i] < labels[j]:
        dw_D[j][:] = dw_D[j][:] + lr*data_T[j][:]
        error += 1

for i in range(len(y_i_D)):
    if y_i_D[i] <= 0:
        y_D[i] = 0
        plt.scatter(data[0][i], data[1][i], c='r')
    else:
        y_D[i] = 1
        plt.scatter(data[0][i], data[1][i], c='b')

w_D = adjust_w(w_D, dw_D)
y_i_D = np.matmul(w_D, data[:3])
y_i_D = np.sum(y_i_D, axis=0)
error_ls_D.append(error/n) 

    # if error == 0:
    #     break

#plt.plot(error_ls, c='r')
plt.plot(error_ls_D, c='b')
plt.show()














