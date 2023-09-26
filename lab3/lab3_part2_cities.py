import numpy as np
import matplotlib.pyplot as plt
import random

#NOTE: DEAD UNIT IS A PROBELM. Isolated points seem to attract multiple nodes where one of the nodes remains dead.
#Adjusting the update depending on how close each neighbor of the winner is to the sample in question is the best solution ive found.
#Adjusting the init weights to our data also seems to help.

#IDE: runna den typ 100 gånger och ta average result, det som funkar för tillfället

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
def init_network(dim, n, data):

    net = np.zeros([n,dim])

    #Init weights from input samples in order to avoid dead units
    for i in range(n):
        #Performance increases significantly when we decrease the range of randomness, obviously maybe I dont know
        random_term = random.uniform(0.6, 1.4)
        net[i] = data[i]*random_term

    #net = np.random.uniform(0, 1, (10, 2))

    return net

#Find the distance to all other points for each point, we use it in order to adjust the w-updates towards each point
def p_density(data):

    dist_ls = np.zeros(len(data))

    for i, point in enumerate(data):
        dist = 0
        for next_point in data:
            dist += np.linalg.norm(point-next_point)
        dist_ls[i] = dist

    return dist_ls

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
def find_hood(net, winner, x, t):

    thr = 2 - int(t*3/20)
    hood = []
    circ_hood = False  

    lb = winner-int(thr/2)
    ub = winner+int(thr/2)

    if thr == 0:
        hood.append(winner)
        return hood

    if ub > len(net)-1:
        ub_a = len(net)-1
        ub_b = ub-len(net)-1
        circ_hood = True

    if circ_hood == True:
        for i in range(lb, ub_a+1):
            hood.append(i)
        for i in range(0, ub_b+1):
            hood.append(i)
    else:
        for i in range(lb, ub+1):
            hood.append(i)

    return hood

#Update weights
def adjust_w(node, x, lr, term):
    
    w_old = node
    w_new = w_old + lr * (x-w_old) / (term)

    return w_new
    
#Training loop
def train(net, data, lr):

    n_epochs = 20
    dist_ls = p_density(data)

    for epoch in range(n_epochs):
        for i, x in enumerate(data):
            win = find_winner(net, x)
            hood = find_hood(net, win, x, epoch)
            for j in hood:
                if j != win:
                    w_new = adjust_w(net[j], x, lr, dist_ls[i])
                else:
                    w_new = adjust_w(net[j], x, lr, 1)
                net[j] = w_new
            
    return net

def results(net, data):

    results = []

    for i, x in enumerate(data):
        win = net[find_winner(net, x)]
        results.append([win, x])

    return results
    

def main():

    #Step-size/learning-rate, 0.2 with decrease turned out to be ok
    lr = 0.1
    coords_ls = np.zeros([10,2])
    n_runs = 15
    data = read_data()

    for i in range(n_runs):
        net = init_network(2, 10, data)
        trained_net = train(net, data, lr)
        for j, p in enumerate(trained_net):
            coords_ls[j] += p

    coords_ls = coords_ls/n_runs

    for i, coord in enumerate(coords_ls):
        plt.scatter(coord[0], coord[1], c='b')
        plt.scatter(data[i][0], data[i][1], c='r')
        plt.plot([coords_ls[i-1][0], coords_ls[i][0]], [coords_ls[i-1][1], coords_ls[i][1]], color='red', linestyle='-', linewidth=2)
    plt.show()

main()