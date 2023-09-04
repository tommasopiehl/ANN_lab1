# CLASSIFICATION WITH SINGLE-LAYER PERCEPTRON
# WITH LINEARLY SEPERABLE DATA 

import numpy as np
import matplotlib.pyplot as plt 

# Constants
LEARNING_RATE = 0.001
EPOCHS = 25

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
    mA = [-3.0, 1]
    sigmaA = 0.5
    mB = [3.0, -1]
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
    #return -(weights[0]+weights[1]*x)/weights[2]
    return -(np.sum(weights[0])+np.sum(weights[2])*x)/np.sum(weights[1])


def plot_decision_boundary(weights, type):
    if(type == "perceptron"): fig = 2
    elif(type == "delta"): fig = 3
    plt.figure(fig)
    x = np.linspace(-6,6,num=6)
    y = decision_boundary(x, weights)
    plt.plot(x,y)
    plt.xlim([-6, 6])
    plt.ylim([-5, 5])
    
    plt.savefig(type+".png")

def plot_error():
    pass

def plot_missclassification(epoch, missclassified):
    plt.figure(4)
    plt.plot([epoch],[missclassified], marker = "o")
    plt.savefig("missclassified.png")

    
def iterative_classification(patterns, targets, weights, type):
    
    # helpful to plot a line and after each epoch of training 
    # iteratively replotting the updated decision boundary 
    # desicion boundary = line where Wx = 0
    # om felklassificeras
    
    patternsT = np.transpose(patterns)
    
    for epoch in range(EPOCHS):
        missclassified = 0
        
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
                if weighted_sum < 0:
                    current_class = 0
                else:
                    current_class = 1
                
                error = targets[pattern_index]-current_class
                
                if error != 0:
                    missclassified +=1
            
            elif type == "delta":
                # detta är typ steepest decent? men i definierade epoker?
                error = targets[pattern_index]-weighted_sum
                
            for i in range(len(weights)):
                if(error == 0 and LEARNING_RATE*error*patternsT[pattern_index][i]!= 0):
                    print("stoppp")
                weights[i] += LEARNING_RATE*error*patternsT[pattern_index][i]
                
            if pattern_index % 20 == 0:
                plot_decision_boundary(weights, type)
                print(epoch)
                
        if type == "perceptron":
            plot_missclassification(epoch, missclassified)

def batch_classification():
    pass 

def main():
    classA, classB = generate_data()
    plot_data(classA, classB, 1)
    patterns, targets, symmetric_targets = create_patterns_targets(classA, classB)
    # skicka in dimensions på vikter 
    weights = initialize_weights(patterns, targets)
    plot_data(classA, classB, 2)
    iterative_classification(patterns, targets, weights, type = "perceptron")
    plot_data(classA, classB, 3)
    iterative_classification(patterns, symmetric_targets, weights, type = "delta")
    
    
   
    


if __name__ == '__main__':
    main()
    
    
# write code flexible in terms of number of
# epochs
epochs = 20 

# training data is stored in:
# dont forget to add an extra row with 
# ones for bias

# flexible input and output patterns size
# and number of traning patterns 

    
# plotta en learning curve ?



