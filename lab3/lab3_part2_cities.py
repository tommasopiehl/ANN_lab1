import numpy as np
import matplotlib.pyplot as plt

#Read data from file
def read_data():

    data = []

    with open("/Users/tommasopiehl/ANN_lab1/lab3/data_lab2/cities.dat", 'r') as file:
        for line in file:
            data.append(line.strip().split(','))
            data[-1][1] = data[-1][1][:-1]

    data_matrix = np.array(data).reshape(10, 2)

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

    thr = 2 - int(3*t/20)
    hood = []
    circ_hood = False

    lb = winner-int(thr)
    ub = winner+int(thr)

    if ub > len(net)-1:
        ub_a = len(net)-1
        ub_b = ub-len(net)-1
        circ_hood = True

    if circ_hood == True:
        for i in range(lb-1, ub_a):
            term = abs(winner-i)
            if term == 0:
                term = 1
            hood.append([i, term])
        for i in range(0, ub_b):
            term = abs(winner-i)
            if term == 0:
                term = 1
            hood.append([i, term])
    else:
        for i in range(lb-1, ub):
            term = abs(winner-i)
            if term == 0:
                term = 1
            hood.append([i, term])

    return hood

#Update weights
def adjust_w(node, x, lr, term):

    #Term/200 seems to be best
    w_old = node
    w_new = w_old + lr * (x-w_old)/(term/200)

    return w_new
    
#Training loop
def train(net, data, lr):

    n_epochs = 30

    for epoch in range(n_epochs):
        #x_i = animal i

        plt_results = results(net, data)
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

def results(net, data):

    results = []

    for i, x in enumerate(data):
        win = find_winner(net, x)
        results.append([win, i])

    results_sorted = sorted(results, key=lambda x: x[0])

    return results_sorted

def main():

    #Step-size/learning-rate, 0.2 tunred out to be very high
    lr = 0.0005

    result = []

    data = read_data()
    net = init_network(2, 10)
    trained_net = train(net, data)

    return result


print(main())