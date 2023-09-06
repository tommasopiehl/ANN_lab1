# CLASSIFICATION WITH SINGLE-LAYER PERCEPTRON
# WITH LINEARLY SEPERABLE DATA 

import numpy as np
import matplotlib.pyplot as plt 

# Constants
LEARNING_RATE = 0.01
EPOCHS = 5000
CONVERGENCE = 0.0001
BIAS = True
np.random.seed(42)



# GENERATE DATA 
def generate_data():
    # code from lab description
    # to generate data 
        
    # choose parameters, mA, mB, sigmaA, sigmaB, yourselves to make sure that
    # the two sets are linearly separable (so the means of the two distributions, mA and
    # mB, should be sufficiently different)
    
    # sigma = standard deviation
    # m = medelvärde
    n = 100
    mA = [-2.0, 1]
    sigmaA = 0.5
    mB = [2.0, -1]
    sigmaB = 0.5

    classA = np.zeros((2, n))
    classB = np.zeros((2, n))
    
    classA[0, :] = np.random.randn(1, n) * sigmaA + mA[0]
    classA[1, :] = np.random.randn(1, n) * sigmaA + mA[1]
    classB[0, :] = np.random.randn(1, n) * sigmaB + mB[0]
    classB[1, :] = np.random.randn(1, n) * sigmaB + mB[1]
    return classA, classB 


def plot_data(classA, classB, fig):
    plt.figure(fig)
    plt.scatter(classA[0,:], classA[1,:], c ="b")
    plt.scatter(classB[0,:], classB[1,:], c ="r")
    plt.title("Generated data")
    #plt.grid
    # plt.savefig("mygraph.png")
    # ta reda på vad som är x-värde och vad som är y ??
    
def create_patterns_targets(classA, classB):
    
    targets_classA = np.ones(len(classA[0]))
    targets_classB = np.zeros(len(classB[0]))
    targets = np.append(targets_classA, targets_classB)
    
    symmetric_targets_classA = np.ones(len(classA[0]))
    symmetric_targets_classB = np.full(len(classB[0]),-1)
    symmetric_targets = np.append(symmetric_targets_classA, symmetric_targets_classB)
    
    ## combining matrixes
    patterns = np.concatenate((classA, classB), axis=1)
    
    ## adding a row of ones at the end
    patterns = np.concatenate((patterns, [np.ones(len(patterns[0]))]), axis = 0)
    
    # shuffle 
    shuffle = np.random.permutation(len(targets))
    patterns = patterns[:, shuffle]
    targets = targets[shuffle]
    symmetric_targets = symmetric_targets[shuffle]
    
    return patterns, targets, symmetric_targets

def initialize_weights(patterns, targets):
    # kanske ta dimensioner som input 
    # initialize weights
    # start with small random numbers 
    # drawn from normal distribution with 
    # zero mean 
    
    # number of rows equal to number of features + 1 for bias
    rows = len(patterns) 
    
    sigma = 0.5
    mean = 0
    
    weights = np.random.randn(rows) * sigma + mean
            
    return weights


def decision_boundary(x, weights):
    if BIAS:
        return -(weights[0]*x+weights[2])/weights[1]
    else:
        return -(weights[0]*x)/weights[1]


def plot_decision_boundary(weights, type, title, fig):
    # if(type == "perceptron"): fig = 2
    # elif(type == "delta"): fig = 3
    plt.figure(fig)
    x = np.linspace(-6,6,num=6)
    y = decision_boundary(x, weights)
    plt.plot(x,y)
    plt.axis([-4, 4, -4, 4])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)

    # Add normal vector
    if BIAS:
        m = -weights[2]/weights[1]
    else: 
        m = 0
    plt.quiver(0, m, weights[0], weights[1], scale=1, scale_units='xy', angles='xy')
    legend = ['Positive Class',
              'Negative class',
              'Decision boundary',
              'Normal vector',
              'Axis over origin']

    
    # # Show adjustment vector
    # if adjustment is not None:
    #     plt.quiver(w[0, 1], w[0, 0] + m, adjustment[0], adjustment[1], scale=1, scale_units='xy', angles='xy',
    #                color='g')
    # else:
    #     legend.remove('Adjustment vector')

    # Add axis over the origin as a dotted line with nothcing
    plt.plot([-4, 4], [0, 0], c='k', ls='--')
    plt.plot([0, 0], [-4, 4], c='k', ls='--')

    # Add legend (size 8, upper left corner)
    plt.legend(legend, loc='upper left', fontsize=8)

    
    #plt.savefig(type+".png")
    plt.savefig(title+".png")

    

def plot_error(x, y, title, nr, epoch):
    
    plt.figure(nr)
    #plt.plot(x,y)
    #lines
    # for i in range(1, epoch):
    #     plt.axvline(i, color='r', linestyle='--', linewidth=0.5)
    plt.plot(y, marker='o', color='r', markersize=1, linewidth=0.8)

    legend = ['Mean Squared Error']
    plt.legend(legend, loc='upper left', fontsize=8)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(title)
    #plt.savefig(title +".png")
    plt.savefig(title +".png")


# def plot_missclassification(epoch, missclassified):
#     plt.figure(4)
#     plt.plot([epoch],[missclassified], marker = "o")
#     plt.savefig("missclassified.png")


def mse(errors, n):
    
    mse = 0
    
    for error in errors:
        mse += error**2
    mse = mse/n
    return mse   
    
    
def iterative_classification(patterns, targets, weights, type):
    
    # helpful to plot a line and after each epoch of training 
    # iteratively replotting the updated decision boundary 
    # desicion boundary = line where Wx = 0
    # om felklassificeras
    
    patternsT = np.transpose(patterns)
    errors = []

    
    for epoch in range(EPOCHS):
        missclassified = 0
        change = 0
        epoch_error = []
        
        # går igenom alla punkter
        for pattern_index in range(len(patternsT)):
                       
            # compute weighted sum
            weighted_sum = 0
            
            # går igenom alla dimesioner
            for dimension_index in range(len(patternsT[pattern_index])):
                weighted_sum += patternsT[pattern_index][dimension_index]*weights[dimension_index]
            
            '''
            Handling of perceptron and delta
            '''
            
            if type == "perceptron":  
                title = "Online mode, Perceptron learning"
                if weighted_sum < 0:
                    current_class = 0
                else:
                    current_class = 1
                
                error = targets[pattern_index]-current_class
                
                if error != 0:
                    missclassified +=1
            
            elif type == "delta":
                
                # missclassified
                if weighted_sum < 0:
                    current_class = -1
                else:
                    current_class = 1
                if targets[pattern_index] != current_class:
                    missclassified +=1
                    
                error = targets[pattern_index]-weighted_sum
                epoch_error.append(error)
                title = "Online mode, Delta learning"
                
            for i in range(len(weights)):
                weights[i] += LEARNING_RATE*error*patternsT[pattern_index][i]
                change += LEARNING_RATE*error*patternsT[pattern_index][i]
        errors.append(mse(epoch_error,len(patternsT)))
        print(missclassified, epoch)
        print(change)
        if abs(change) < 0.0001:
            print(epoch," converged")

            break
        # if epoch>1:
        #     if abs(errors[-1]-errors[-2])< CONVERGENCE:
        #         print(epoch," converged")
        #         break 
    error_title = "MSE per Epoch in sequential mode, Learning Rate "+str(LEARNING_RATE)
    plot_error(np.linspace(0,EPOCHS,num=EPOCHS), errors, error_title, 7, epoch)
    plot_decision_boundary(weights, type, title, 11)
                
        # if type == "perceptron":
        #     plot_missclassification(epoch, missclassified)

def batch_classification(patterns, targets, weights):
    errors = []
    patternsT = np.transpose(patterns)

    for epoch in range(EPOCHS):
        
        error = targets-np.dot(weights, patterns[:])
        
        weight_updates = LEARNING_RATE*np.dot(error, patternsT[:])
       
        weights += weight_updates
        
        errors.append(mse(error,len(targets)))
        # print(weights)
        # print(weight_updates)
        # print(abs(sum(weight_updates)))
        if abs(sum(weight_updates)) < 0.0001:
            print(epoch," converged")
            break
        # # if epoch>1:
        #     print(errors[-1])
        #     print(error[-2])
        #     print(errors[-1]-errors[-2])
        #     print(abs(errors[-1]-errors[-2]))
        #     if abs(errors[-1]-errors[-2])< CONVERGENCE:
        #         print(epoch," converged")
        #         break 
    error_title = "MSE per Epoch in batch mode, Learning Rate "+str(LEARNING_RATE)
    plot_error(np.linspace(0,EPOCHS,num=EPOCHS), errors, error_title, 8, epoch)
    plot_decision_boundary(weights, "delta", "Batch mode, Delta learning", 10)
            

def main():
    classA, classB = generate_data()
    plot_data(classA, classB, 1)
    patterns, targets, symmetric_targets = create_patterns_targets(classA, classB)
    # skicka in dimensions på vikter 
    weights = initialize_weights(patterns, targets)
    # plot_data(classA, classB, 2)
    # iterative_classification(patterns, targets, weights, type = "perceptron")
    # plot_data(classA, classB, 11)
    # iterative_classification(patterns, symmetric_targets, weights, type = "delta")
    plot_data(classA, classB, 10)
    batch_classification(patterns, symmetric_targets, weights)

if __name__ == '__main__':
    main()
    
 


