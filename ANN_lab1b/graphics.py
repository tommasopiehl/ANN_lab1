import matplotlib.pyplot as plt
import numpy as np
from lab_1b_attempt1 import forward_pass


def classify(out):
    classes = []
    for value in out[0]:
        if value<0:
            classes.append(-1)
        elif value >= 0:
            classes.append(1)
            
    return classes
    

def plot_decision_boundary(w, v, type, fig, patterns, targets):
    plt.figure(fig)

    #patterns = np.transpose(patterns)
    x_min, x_max = min(patterns[:, 0]) - 1, max(patterns[:, 0]) + 1
    y_min, y_max = min(patterns[:, 1]) - 1, max(patterns[:, 1]) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Use the trained neural network to make predictions for the grid points
    out, hout = forward_pass(grid_points,w, v)
    classes = np.array([classify(out)])
    

    # Reshape predictions to match the grid shape
    classes = classes.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, classes, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(patterns[:, 0], patterns[:, 1], c=targets, cmap=plt.cm.Spectral)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary, 5 hidden nodes')
    plt.savefig(type+".png")






    # # if(type == "perceptron"): fig = 2
    # # elif(type == "delta"): fig = 3
    # plt.figure(fig)
    # x = np.linspace(-6,6,num=6)
    # y = decision_boundary(x, weights)
    # plt.plot(x,y)
    # plt.axis([-4, 4, -4, 4])
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.title(title)

    # # Add normal vector
    # if BIAS:
    #     m = -weights[2]/weights[1]
    # else: 
    #     m = 0
    # plt.quiver(0, m, weights[0], weights[1], scale=1, scale_units='xy', angles='xy')
    # legend = ['Positive Class',
    #           'Negative class',
    #           'Decision boundary',
    #           'Normal vector',
    #           'Axis over origin']

    
    # # # Show adjustment vector
    # # if adjustment is not None:
    # #     plt.quiver(w[0, 1], w[0, 0] + m, adjustment[0], adjustment[1], scale=1, scale_units='xy', angles='xy',
    # #                color='g')
    # # else:
    # #     legend.remove('Adjustment vector')

    # # Add axis over the origin as a dotted line with nothcing
    # plt.plot([-4, 4], [0, 0], c='k', ls='--')
    # plt.plot([0, 0], [-4, 4], c='k', ls='--')

    # # Add legend (size 8, upper left corner)
    # plt.legend(legend, loc='upper left', fontsize=8)

    
    # #plt.savefig(type+".png")
    # plt.savefig("complicated_boundary"+".png")

    


def plot_data(classA, classB, fig, filename):
    plt.figure(fig)
    plt.scatter(classA[0,:], classA[1,:], c ="b", label = "Positive Class")
    plt.scatter(classB[0,:], classB[1,:], c ="r", label = "Negative Class")
    plt.title("Generated data")
    plt.axis([-4, 4, -4, 4])
    
    # Add axis over the origin as a dotted line with nothcing
    plt.plot([-4, 4], [0, 0], c='k', ls='--', label = "Axis over orgin")
    plt.plot([0, 0], [-4, 4], c='k', ls='--')

    plt.legend(loc='upper left', fontsize=8)
    
    plt.savefig(filename)

def plot_error(y, title, nr, c, label):
    
    plt.figure(nr)
    plt.plot(y, marker='o', color=c, markersize=1, linewidth=0.8, label = label)

    plt.legend(loc='upper left', fontsize=8)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title(title)
    #plt.savefig(title +".png")
    plt.savefig(title +".png")

    
  
