import numpy as np
import matplotlib.pyplot as plt
import random

def read_data():

    data = []

    with open("/Users/tommasopiehl/ANN_lab1/lab3/data_lab2/votes.dat", 'r') as file:
        data = file.read().rstrip().split(',')

    data_matrix = np.array(data).reshape(349, 31)

    return data_matrix.astype(float)


#Init SOM-network with w-values between 0 and 1
def init_network(dim, n):

    net = np.random.rand(n, dim)

    return net

#Find winner for CL
def find_winner(net, x):

    best_dist = np.inf
    winner = -1

    for i, node in enumerate(net):
        dist = np.sqrt(np.dot((x-node).T, (x-node)))
        if dist < best_dist:
            best_dist = dist
            winner = i

    return winner

#Find neighborhood
def find_hood(net, winner, t):

    thr = int(50-t*2.6)

    lb = max(0, winner-thr)
    ub = min(len(net)-1, winner+thr)

    hood = []

    if lb-ub == 0:
        hood = [[winner, 1]]
        return hood

    for i in range(lb-1, ub+1):
        if i >= 0:
            
            #Term is used in order to update the neighbors with regards to their distance to the winner
            term = abs(winner-i)
            if term == 0:
                term = 1
            hood.append([i, term])

    return hood

#Update weights
def adjust_w(node, x, lr, term):

    #We dont use term anymore
    w_old = node
    w_new = w_old + lr * (x-w_old)

    return w_new

#Training loop
def train(net, data, lr):

    n_epochs = 30

    data_not, names = read_data()

    for epoch in range(n_epochs):
        #x_i = animal i

        plt_results = results(net, data, names)
        for i,res in enumerate(plt_results):
            plt.scatter(res[0], 0, c ='r')
            plt.text(res[0], 0.001*i, res[1], fontsize=8)

        plt.show()

        for i, x in enumerate(data):
            win = find_winner(net, x)
            hood = find_hood(net, win, epoch)
            for j, term in hood:
                w_new = adjust_w(net[j], x, lr, term)
                net[j] = w_new

    return net

def results(net, data, names):

    results = []

    for i, x in enumerate(data):
        win = find_winner(net, x)
        results.append([win, names[i]])

    results_sorted = sorted(results, key=lambda x: x[0])

    return results_sorted

def main():

    #Step-size/learning-rate
    lr = 0.2

    data, names = read_data()
    net = init_network(10, 10)
    trained_net = train(net, data, lr)
    result = results(trained_net, data, names)

    return result


print(main())