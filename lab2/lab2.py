import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

n = 500
mA = [1, 1]
sigmaA = 0.5
mB = [-0.5, -0,5]
sigmaB = 0.5
alpha = 0.9
lr = 0.01
BIAS = True

# GENERATE DATA 
def generate_data_train():

    classA = np.zeros((2, int(n/2)))
    classB = np.zeros((2, int(n/2)))
    
    classA[0, :] = np.random.randn(1, int(n/2)) * sigmaA + mA[0]
    classA[1, :] = np.random.randn(1, int(n/2)) * sigmaA + mA[1]
    classB[0, :] = np.random.randn(1, int(n/2)) * sigmaB + mB[0]
    classB[1, :] = np.random.randn(1, int(n/2)) * sigmaB + mB[1]

    pA = 0.5
    pB = 0.0

    # Calculate the number of points to extract
    num_A = int(len(classA.T) * pA)
    num_B = int(len(classB.T) * pB)

    labels_train = np.zeros(num_A+num_B)
    labels_val = np.zeros(n-(num_A+num_B))

    # Randomly select the indices of the points to extract
    indx_A = np.random.choice(len(classA.T), num_A, replace=False)
    indx_B = np.random.choice(len(classB.T), num_B, replace=False)

    # Extract the selected points
    points_A = classA.T[indx_A]
    points_B = classB.T[indx_B]

    left_A = np.delete(classA.T, indx_A, axis=0)
    left_B = np.delete(classB.T, indx_B, axis=0)

    train = np.ones([3, num_A+num_B])
    train[0] = np.concatenate((points_A.T[0], points_B.T[0]))
    train[1] = np.concatenate((points_A.T[1], points_B.T[1]))

    val = np.ones([3, n-(num_A+num_B)])
    val[0] = np.concatenate((left_A.T[0], left_B.T[0]))
    val[1] = np.concatenate((left_A.T[1], left_B.T[1]))

    np.apply_along_axis(np.random.shuffle, axis=1, arr=train)
    np.apply_along_axis(np.random.shuffle, axis=1, arr=val)

    for i, sample in enumerate(train[0]):
        if sample in classA[0]:
            labels_train[i] = 1
            plt.scatter(sample, train[1][i], c='b')
        elif sample in classB[0]:
            labels_train[i] = -1
            plt.scatter(sample, train[1][i], c='r')
    
    plt.show()
    
    for i, sample in enumerate(val[0]):
        if sample in classA[0]:
            labels_val[i] = 1
            plt.scatter(sample, val[1][i], c='b')
        elif sample in classB[0]:
            labels_val[i] = -1
            plt.scatter(sample, val[1][i], c='r')

    plt.show()

    return train, val, labels_train, labels_val

def generate_data():

    patterns = np.ones([3, n])
    classA = np.zeros((2, int(n/2)))
    classB = np.zeros((2, int(n/2)))
    
    classA[0, :] = np.random.randn(1, int(n/2)) * sigmaA + mA[0]
    classA[1, :] = np.random.randn(1, int(n/2)) * sigmaA + mA[1]
    classB[0, :] = np.random.randn(1, int(n/2)) * sigmaB + mB[0]
    classB[1, :] = np.random.randn(1, int(n/2)) * sigmaB + mB[1]

    patterns[0]= np.concatenate((classA[0], classB[0]))
    patterns[1]= np.concatenate((classA[1], classB[1]))
    np.apply_along_axis(np.random.shuffle, axis=1, arr=patterns)

    labels = np.zeros(n)

    for point in classA.T:
        plt.scatter(point[0], point[1], c='b')
    for point in classB.T:
        plt.scatter(point[0], point[1], c='r')
    plt.show()

    for i, sample in enumerate(patterns[0]):
        if sample in classA[0]:
            labels[i] = 1
        elif sample in classB[0]:
            labels[i] = -1
    
    return patterns, labels, classA, classB


#INIT W
def initialize_W(patterns):

    rows = len(patterns) 
    
    sigma = 0.5
    mean = 0
    
    weights = np.random.randn(hidden_size, rows) * sigma + mean
            
    return weights

#INIT V
def initialize_V():

    rows = hidden_size 
    
    sigma = 0.5
    mean = 0
    
    weights = np.random.randn(output_size, rows+1) * sigma + mean
            
    return weights

def transfer(x):

    return 2/(1+np.exp(-x))-1

def validate(data, labels, w, v):

    h_sums = np.dot(w, data)
    h = transfer(h_sums)
    h = np.concatenate((h, np.ones([1,len(h[0])])), axis=0)
    o_sums = np.dot(v, h)
    O = transfer(o_sums)

    miss = 0
    for i in range(len(data[0])):
        if O[0][i] != labels[i]:
            miss += 1

    return miss/n


#25: miss= 26
#20: miss= 25
#15: miss= 27
#10: miss= 18
#5: miss= 23

hidden_size = 10
output_size = 1
data, labels, classA, classB = generate_data()
#train, val, labels_train, labels_val = generate_data_train()
n_epochs = 500

#main part 1
def main():

    # data = train
    # labels = labels_train
    w = initialize_W(data)
    v = initialize_V()
    error_ls = []
    conv_epoch = -1

    for epoch in range(n_epochs):

        #forward-pass
        h_sums = np.dot(w, data)
        h = transfer(h_sums)
        h = np.concatenate((h, np.ones([1,len(h[0])])), axis=0)
        o_sums = np.dot(v, h)
        O = transfer(o_sums)

        #backward-pass
        delta_o = (O-labels)*((1+O)*(1-O)/2)
        delta_h = np.dot(v.T, delta_o)*((1+h)*(1-h)/2)

        #weight update
        dw = -lr*np.dot(delta_h, data.T)
        dv = -lr*np.dot(delta_o, h.T)

        dw_m = alpha*dw - (1-alpha)*np.dot(delta_h, data.T)
        dv_m = alpha*dv - (1-alpha)*np.dot(delta_o, h.T)

        w += dw_m[:hidden_size]
        v += dv_m

        miss = 0
        y = np.zeros(len(labels))
        for i, pred in enumerate(O[0]):
            if pred < 0:
                y[i] = -1
                if labels[i] != -1:
                    miss += 1
            else:
                y[i] = 1
                if labels[i] != 1:
                    miss += 1

        print("train", miss)
        error_ls.append(miss/n)

        if abs(np.sum(dw)) + abs(np.sum(dv)) < 0.00005:
            conv_epoch = epoch
            break

    if abs(np.sum(dw)) + abs(np.sum(dv)) > 0.00005:
        conv_epoch = n_epochs-1

    print(conv_epoch)
    plt.plot(np.linspace(0,conv_epoch+1,num=conv_epoch+1), error_ls)
    plt.show()

    return 0

        
main()