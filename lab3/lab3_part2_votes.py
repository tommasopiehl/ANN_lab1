import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
import matplotlib.transforms as transforms
from matplotlib.colors import ListedColormap
#Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
#Use some color scheme for these different groups
#I have adapted the coding of the parties to better represent the left-right scale in politics
#New coding: 0=no party, 1='v', 2='mp', 3='s', 4='c', 5='fp', 6='m', 7='kd'

#Coding: Male 0, Female 1

#NOTES ON RESULT: We see similar patterns for all the categories, seems like a lot of women vote in the middle,
#districts 

def read_data():

    data = []
    data_parties = []
    data_distr = []
    data_sex = []

    with open("/Users/tommasopiehl/ANN_lab1/lab3/data_lab2/votes.dat", 'r') as file:
        data = file.read().rstrip().split(',')

    with open('/Users/tommasopiehl/ANN_lab1/lab3/data_lab2/mpdistrict.dat', 'r') as file:
        for line in file:
            data_distr.append(int(line))
    
    with open('/Users/tommasopiehl/ANN_lab1/lab3/data_lab2/mpsex.dat', 'r') as file:
        for line in file:
            data_sex.append(int(line))

    with open('/Users/tommasopiehl/ANN_lab1/lab3/data_lab2/mpparty.dat', 'r') as file:
        for line in file:
            new_line = line
            if int(line) == 1:
                new_line = 6
            if int(line) == 2:
                new_line = 5
            if int(line) == 3:
                new_line = 3
            if int(line) == 4:
                new_line = 1
            if int(line) == 5:
                new_line = 2
            if int(line) == 6:
                new_line = 7
            if int(line) == 7:
                new_line = 4
            data_parties.append(int(new_line))

    data_matrix = np.array(data).reshape(349, 31)

    return data_matrix.astype(float), data_parties, data_distr, data_sex

#Init SOM-network with w-values between 0 and 1
def init_network(dim, n):

    net = np.random.rand(n, dim)
    net_grid = []

    for i in range(10):
        net_grid.append([])
    
    for i in range(len(net)):
        net_grid[i%10].append(net[i])

    return net, net_grid

#Find winner for CL
def find_winner(net, x):

    best_dist = np.inf
    winner = -1

    for i in range(10):
        for j in range(10):
            dist = np.sqrt(np.dot((x-net[i][j]).T, (x-net[i][j])))
            if dist < best_dist:
                best_dist = dist
                winner = [i,j]

    return winner

#Find neighborhood
def find_hood(net, winner, t):

    thr = int(2-t/10)

    hood = []

    for i in range(10):
        for j in range(10):
            if abs(winner[0] - i) + abs(winner[1] - j) <= thr:
                hood.append([i,j])

    return hood

#Update weights
def adjust_w(node, x, lr):

    w_old = node
    w_new = w_old + lr * (x-w_old)

    return w_new

#Training loop
def train(net, data, lr):

    n_epochs = 20

    for epoch in range(n_epochs):

        for i, x in enumerate(data):
            win = find_winner(net, x)
            hood = find_hood(net, win, epoch)
            for indx in hood:
                w_new = adjust_w(net[indx[0]][indx[1]], x, lr)
                net[indx[0]][indx[1]] = w_new

    return net

def avg_grid(net, data, parties):

    avg_winners = np.zeros([10,10])
    avg_count = np.zeros([10,10])

    for i, x in enumerate(data):
        win = find_winner(net, x)
        avg_winners[win[0]][win[1]] += parties[i]
        avg_count[win[0]][win[1]] += 1

    for i in range(10):
        for j in range(10):
            if type(avg_winners[i][j]/avg_count[i][j]) == float:
                avg_winners[i][j] = round(avg_winners[i][j]/avg_count[i][j])
            else:
                avg_winners[i][j] = 0

    return avg_winners

def plot_grid(result, decode_ls):

    party_colours = {
    0: "black",  # no party
    1: (36/255, 168/255, 255/255),  # light blue - moderates
    2: (36/255, 50/255, 255/255),  # dark blue - liberals
    3: (255/255, 58/255, 36/255),  # red - social democrats
    4: (192/255, 41/255, 25/255),  # dark red - left
    5: (24/255, 156/255, 3/255),  # dark green - greens
    6: (0/255, 0/255, 129/255),  # dark blue - christian democrats
    7: (0/255, 246/255, 41/255)  # light green - center
    }

    cmap = ListedColormap(party_colours)
    color_grid = cmap(result)

    # if len(decode_ls) == 2:
    #     cmap_data = np.ones([10, 10])*-0.2
    # else:
    #     cmap_data = np.ones([10, 10])*-1

    # for i in range(10):
    #     for j in range(10):
    #         if result[i][j] >= 0:
    #             cmap_data[i][j] = result[i][j]

    plt.imshow(result, cmap=color_grid, interpolation='none', aspect='auto')
    plt.colorbar()
    plt.show()

def main():

    #Step-size/learning-rate
    lr = 0.2

    data, parties, distr, gender = read_data()
    net, net_grid = init_network(31, 100)
    trained_net = train(net_grid, data, lr)
    result = avg_grid(trained_net, data, parties)
    result_distr = avg_grid(trained_net, data, distr)
    result_gender = avg_grid(trained_net, data, gender)

    decode_ls = ['None','V','MP','S','C','FP','M','KD']
    decode_ls_gender = ['Man','Woman']

    plot_grid(result, decode_ls=decode_ls)
    plot_grid(result_gender, decode_ls=decode_ls)
    plot_grid(result_distr, decode_ls=decode_ls)

    return trained_net

main()


# def plot_result(result, category, decode_ls=None):

#     # cmap_data = np.zeros([100,2])
#     # cmap_id = np.zeros(100)
#     # avg_pos = np.zeros([len(decode_ls), 2])
#     # party_count = np.zeros(8)

#     #Normalize nerouns in positive quadrant
#     # for res in result:
#     #     coords = res[0]
#     #     coords[0] += 1.5
#     #     coords[1] += 1.5
#     #     coords[0] *= 0.9

#     cell_index = 0
#     for i in np.linspace(0,3,10):
#         for j in np.linspace(0,3,10):
#             cmap_data[cell_index] = np.array([i,j])
#             cell_index += 1

#     #Compute the averge index for each cell from every neuron that has a distance of < 0.5
#     for i, cell in enumerate(cmap_data):
#         avg_id = 0
#         id_count = 0
#         for res in result:
#             party_id = res[1]
#             coords = res[0]
#             dist = np.linalg.norm(cell-coords)
#             if dist <= 0.5:
#                 avg_id += party_id
#                 id_count += 1
#         if id_count > 0:
#             cmap_id[i] = avg_id/id_count

#     if len(decode_ls) > 0:
#     #Plot the actual neurons
#         for res in result:
#             party_id = res[1]
#             coords = res[0]
#             avg_pos[party_id] += coords
#             party_count[party_id] += 1
#             plt.scatter(coords[0], coords[1], c='b')
#             plt.text(coords[0],coords[1],decode_ls[party_id],fontsize=5, color='red')
#     else:
#         for res in result:
#             id = res[1]
#             coords = res[0]
#             plt.scatter(coords[0], coords[1], c='b')
#             plt.text(coords[0],coords[1],id,fontsize=5, color='red')

#     plt.title("Positions of each neuron and the sample which it represents, "+str(category))
#     plt.show()

#     if len(decode_ls) > 0:
#         for i in range(len(avg_pos)):
#             avg_pos[i] = avg_pos[i]/party_count[i]
#             plt.scatter(avg_pos[i][0],avg_pos[i][1])
#             plt.text(avg_pos[i][0],avg_pos[i][1],decode_ls[i],fontsize=25, color='black')

#     plt.title("Average position of each, "+str(category))
#     plt.show()

#     #10x10 grid colormap showing the precense of each party
#     cmap_id_plt = cmap_id.reshape(10,10).T
#     fig, ax = plt.subplots()
#     cmap = plt.get_cmap('viridis')
#     im = ax.imshow(cmap_id_plt, cmap=cmap)

#     #For rotating colormap
#     rotation_angle_degrees = 0
#     rotation_transform = transforms.Affine2D().rotate_deg(rotation_angle_degrees)
#     im.set_transform(rotation_transform + ax.transData)
#     ax.set_xlim(0, 8)
#     ax.set_ylim(0, 8)
#     cbar = fig.colorbar(im)
#     plt.title("10x10 colormap to show the average value of, "+str(category))
#     plt.show()