from pywaffle import Waffle
import numpy as np
import matplotlib.pyplot as plt

# import data
from ANN_lab2.SOM_3_3 import SOM

votes_data = np.loadtxt("data_lab2/votes.dat", delimiter=",")

# reshape into 349 samples of 31 features
votes_data = votes_data.reshape(349, -1)

party_colours = {
    0: "white",  # no party
    1: (36 / 255, 168 / 255, 255 / 255),  # light blue - moderates
    2: (36 / 255, 50 / 255, 255 / 255),  # dark blue - liberals
    3: (255 / 255, 58 / 255, 36 / 255),  # red - social democrats
    4: (192 / 255, 41 / 255, 25 / 255),  # dark red - left
    5: (24 / 255, 156 / 255, 3 / 255),  # dark green - greens
    6: (0 / 255, 0 / 255, 129 / 255),  # dark blue - christian democrats
    7: (0 / 255, 246 / 255, 41 / 255)  # light green - center
}

# Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
# Use some color scheme for these different groups

# import labels
party_labels = np.loadtxt("data_lab2/mpparty.dat", delimiter="\n")

# Coding: Male 0, Female 1
sex_labels = np.loadtxt("data_lab2/mpsex.dat", delimiter="\n")

# Coding Districts are just numbers 1-29
district_labels = np.loadtxt("data_lab2/mpdistrict.dat", delimiter="\n")

# Names of the MPs
names_labels = np.loadtxt("data_lab2/mpnames.txt", dtype=str, delimiter="\n", encoding="latin-1")

party_labels = party_labels.reshape(-1, 1)
sex_labels = sex_labels.reshape(-1, 1)
district_labels = district_labels.reshape(-1, 1)
names_labels = names_labels.reshape(-1, 1)

# Map the votes with SOM

# Hyperparameters
epochs_som = 300
eta_som = 0.005
sigma_min = 0.1
sigma_max = 8.0
hidden_shape = (13, 7)
n_features = 31

som = SOM(hidden_shape, dimension=n_features, random=True)

som.train(votes_data.T, epochs_som, eta_som, sigma_min, sigma_max, "votes_som.gif", save_gif=False)

# find the node for each MP
nodes = []

for i, vote in enumerate(votes_data):
    node = som.get_winner(vote)
    nodes.append(node)

# plot the findings

tile_dict = {}
for i in range(hidden_shape[0]):
    for j in range(hidden_shape[1]):
        tile_dict[(i, j)] = []

for i, node in enumerate(nodes):
    tile_dict[node].append((party_labels[i], names_labels[i], sex_labels[i], district_labels[i]))

party_color_grid = {}
party_district_grid = {}
party_sex_grid = {}

for i in range(hidden_shape[0]):
    for j in range(hidden_shape[1]):
        if len(tile_dict[(i, j)]) == 0:
            party_color_grid[(i, j)] = "white"
            party_district_grid[(i, j)] = ""
            party_sex_grid[(i, j)] = -1
        else:
            tile_sex = []
            tile_party = []
            tile_district = []
            for item in tile_dict[(i, j)]:
                tile_sex.append(item[2])
                tile_party.append(item[0])
                tile_district.append(item[3])

            # average the values for gender
            party_sex_grid[(i, j)] = np.average(np.array(tile_sex))

            # find the most common party
            party, counts = np.unique(np.array(tile_party), return_counts=True)
            party_color_grid[(i, j)] = party_colours[party[np.argmax(counts)]]

            # find the most common district
            district, counts = np.unique(np.array(tile_district), return_counts=True)
            party_district_grid[(i, j)] = district[np.argmax(counts)]



# Plot the party colors in a waffle grid
fig, ax = plt.subplots(figsize=(16, 4))
s_size = 1000
for i in range(hidden_shape[0]):
    for j in range(hidden_shape[1]):
        ax.scatter(i, j, color=party_color_grid[(i, j)], s=s_size, marker="s")
ax.set_title("Party colors in a grid")
ax.set_axis_off()
plt.show()


input("Plotted big grid. Press enter to continue...")


# Make plot with 6 subplots, where every other subplot is a legend
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
s_size = 600

# make upper row of subplots square
axs[0, 0].set_aspect("equal")
axs[0, 1].set_aspect("equal")
axs[0, 2].set_aspect("equal")

# make lower row of subplots 1:5
axs[1, 0].set_aspect("equal")
axs[1, 1].set_aspect("equal")
axs[1, 2].set_aspect("equal")

# plot the party colors in a grid
for i in range(hidden_shape[0]):
    for j in range(hidden_shape[1]):
        axs[0, 0].scatter(i, j, color=party_color_grid[(i, j)], s=s_size)
axs[0, 0].set_title("Party colors in a grid")

# plot the districts in a grid
for i in range(hidden_shape[0]):
    for j in range(hidden_shape[1]):
        # put numbers in the middle of the tiles
        axs[0, 1].scatter(i, j, color="white", s=s_size)
        axs[0, 1].text(i, j, party_district_grid[(i, j)], ha="center", va="center", color="black", fontsize=10)

axs[0, 1].set_title("Districts in a grid")

# plot gender in a grid
for i in range(hidden_shape[0]):
    for j in range(hidden_shape[0]):

        if party_sex_grid[(i, j)] < 0:
            r, g, b = 255, 255, 255

        else:
            r = 255 + (53 - 255) * party_sex_grid[(i, j)]
            g = 53 + (93 - 53) * party_sex_grid[(i, j)]
            b = 53 + (255 - 53) * party_sex_grid[(i, j)]

        axs[0, 2].scatter(i, j, color=(r / 255, g / 255, b / 255), s=s_size)

# plot the party colors and the name of the party on a line
party_names = ["no party", "M", "FP", "S", "V", "MP", "KD", "C"]

for i in range(8):
    if i != 0:
        axs[1, 0].scatter(i, 0, color=party_colours[i], s=s_size)
        axs[1, 0].scatter(i, -1, color="white", s=s_size)
        axs[1, 0].text(i, -1, party_names[i], ha="center", va="center", color="black", fontsize=20)
        for j in range(1,7):
            axs[1, 0].scatter(i, -j, color="white", s=s_size)

# Show the gradient from male to female in a plot and write names at either end
x = np.linspace(0, 1, 100)
r = 255 + (53 - 255) * x
g = 53 + (93 - 53) * x
b = 53 + (255 - 53) * x


for i in range(100):
    axs[1, 2].scatter(x[i], 0, color=(r[i] / 255, g[i] / 255, b[i] / 255), s=s_size)
    axs[1, 2].scatter(x[i], -1, color="white", s=s_size)
# write names at either end
axs[1, 2].text(0, -0.2, "Male", ha="left", va="center", color="black", fontsize=20)
axs[1, 2].text(1, -0.2, "Female", ha="right", va="center", color="black", fontsize=20)

# remove the axis labels and axis from lower row
for i in range(3):
    axs[1, i].set_axis_off()


plt.show()
