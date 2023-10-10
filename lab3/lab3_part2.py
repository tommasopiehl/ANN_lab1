import numpy as np
import matplotlib.pyplot as plt

#Read data from file
def read_data():

    data_names = []

    with open("/Users/tommasopiehl/ANN_lab1/lab3/data_lab2/animals.dat", 'r') as file:
        data = file.read().rstrip().split(',')

    with open('/Users/tommasopiehl/ANN_lab1/lab3/data_lab2/animalnames.txt', 'r') as file:
        for line in file:
            data_names.append(line.rstrip()[1:-1])

    data_matrix = np.array(data).reshape(32, 84)

    return data_matrix.astype(int), data_names

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

    thr = int(50-t*2.5)

    lb = max(0, winner-thr)
    ub = min(len(net)-1, winner+thr)

    hood = []

    if lb-ub == 0:
        hood.append(winner)
        return hood

    for i in range(lb-1, ub+1):
        if i >= 0:
            hood.append(i)

    return hood

#Update weights
def adjust_w(node, x, lr):

    w_old = node
    w_new = w_old + lr * (x-w_old)

    return w_new
    
#Training loop
def train(net, data, lr):

    n_epochs = 20

    data_not, names = read_data()

    for epoch in range(n_epochs):
        #x_i = animal i

        plt_results = results(net, data, names)
        plt_y = {}
        for i, res in enumerate(plt_results):
            plt.scatter(res[0], 0, c ='r')
            if i % 2 == 0:
                plt.text(res[0], 0.001*i, res[1], fontsize=10)

        plt.title("Epoch: "+str(epoch))
        plt.xlim([0, 110])
        plt.ylim([-0.02, 0.05])
        plt.show()

        for i, x in enumerate(data):
            win = find_winner(net, x)
            hood = find_hood(net, win, epoch)
            for j in hood:
                w_new = adjust_w(net[j], x, lr)
                net[j] = w_new

    return net

def results(net, data, names):

    results = []

    for i, x in enumerate(data):
        win = find_winner(net, x)
        results.append([win, names[i]])

    results_sorted = sorted(results, key=lambda x: x[0])

    win_count = np.zeros(100)
    for res in results_sorted:
        win_count[res[0]] += 1

    print(win_count)

    return results_sorted

def main():

    #Step-size/learning-rate
    lr = 0.2

    data, names = read_data()
    net = init_network(84, 100)
    trained_net = train(net, data, lr)
    result = results(trained_net, data, names)

    return result


print(main())