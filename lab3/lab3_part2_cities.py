import numpy as np
import matplotlib.pyplot as plt
import random

#NOTE: DEAD UNIT IS A PROBELM. Isolated points seem to attract multiple nodes where one of the nodes remains dead.
#Adjusting the update depending on how close each neighbor of the winner is to the sample in question is the best solution ive found.
#Adjusting the init weights to our data also seems to help.

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
        random_term = random.uniform(0.7, 1.3)
        net[i] = data[i]*random_term

    #net = np.random.uniform(0, 1, (10, 2))

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
def find_hood(net, winner, x, t):

    thr = 2 - int(t*3/20)
    hood = []
    circ_hood = False

    lb = winner-int(thr/2)
    ub = winner+int(thr/2)

    if thr == 0:
        hood.append([winner, 1])
        return hood

    if ub > len(net)-1:
        ub_a = len(net)-1
        ub_b = ub-len(net)-1
        circ_hood = True

    if circ_hood == True:
        for i in range(lb, ub_a+1):
            term = np.linalg.norm(x-net[i])
            if term == 0:
                term = 1
            hood.append([i, term])
        for i in range(0, ub_b+1):
            term = np.linalg.norm(x-net[i])
            if term == 0:
                term = 1
            hood.append([i, term])
    else:
        for i in range(lb, ub+1):
            term = np.linalg.norm(x-net[i])
            if term == 0:
                term = 1
            hood.append([i, term])

    return hood

#Update weights
def adjust_w(node, x, lr, term):
    
    w_old = node
    w_new = w_old + lr * (x-w_old) / term

    return w_new
    
#Training loop
def train(net, data, lr, mult):

    n_epochs = 20

    for epoch in range(n_epochs):
        lr -= epoch/1000
        for i, x in enumerate(data):
            win = find_winner(net, x)
            hood = find_hood(net, win, x, epoch)
            for j, term in hood:
                w_new = adjust_w(net[j], x, lr, term*mult)
                net[j] = w_new
            
    return net

def results(net, data):

    results = []

    for i, x in enumerate(data):
        win = net[find_winner(net, x)]
        results.append([win, x])

    return results

#To measure performance over multiple runs
def error(results):

    dist = 0
    win_count = []

    for res in results:
        dist += np.linalg.norm(res[0]-res[1])
        for w in win_count:
            if np.linalg.norm(res[0]-w)==0:
                dist += 100
        win_count.append(res[0])

    return dist
    

def main():

    #Step-size/learning-rate, 0.2 turned out to be ok
    lr = 0.55
    err_ls = []

    for i in range(100):
        data = read_data()
        net = init_network(2, 10, data)
        trained_net = train(net, data, lr, (20))
        res = results(trained_net, data)
        err =  error(res)
        err_ls.append(err)
    
    count = 0
    for e in err_ls:
        if e < 100:
            count += 1

    print(count)

main()