# CLASSIFICATION WITH SINGLE-LAYER PERCEPTRON
# WITH LINEARLY SEPERABLE DATA 
import random
import numpy as np
import matplotlib.pyplot as plt 

# Constants
LEARNING_RATE = 0.001
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
    # n = 100
    # mA = [1, 1.5]
    # #mA = [0.5, 1]

    # sigmaA = 0.5
    # mB = [-1, -1.5]
    # #mB = [-0.5, -1]
    # sigmaB = 0.5

    # classA = np.zeros((2, n))
    # classB = np.zeros((2, n))
    
    # classA[0, :] = np.random.randn(1, n) * sigmaA + mA[0]
    # classA[1, :] = np.random.randn(1, n) * sigmaA + mA[1]
    # classB[0, :] = np.random.randn(1, n) * sigmaB + mB[0]
    # classB[1, :] = np.random.randn(1, n) * sigmaB + mB[1]
    # return classA, classB 
#==========================
    ndata = 100
    mA = [1.0, 0.3]
    sigmaA = 0.2
    mB = [0.0, -0.1]
    sigmaB = 0.3

    classA = np.zeros((2, ndata))
    classA[0, :round(0.5 * ndata)] = np.random.randn(1, round(0.5 * ndata)) * sigmaA - mA[0]
    classA[1, :round(0.5 * ndata)] = np.random.randn(1, round(0.5 * ndata)) * sigmaA + mA[0]
    classA[1, round(0.5 * ndata):] = np.random.randn(1, round(0.5 * ndata)) * sigmaA + mA[1]

    classB = np.zeros((2, ndata))
    classB[0, :] = np.random.randn(1, ndata) * sigmaB + mB[0]
    classB[1, :] = np.random.randn(1, ndata) * sigmaB + mB[1]
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
    
    classAT = np.transpose(classA)
    classBT = np.transpose(classB)
    
    # remove 25% from each class
    # classA = np.transpose(random.choices(classAT, k=int(len(classAT)*0.75)))
    # classB = np.transpose(random.choices(classBT, k=int(len(classBT)*0.75)))
    
    # remove 50 from classA
    # classA = np.transpose(random.choices(classAT, k=int(len(classAT)*0.5)))
    
    # remove 50 from B
    classB = np.transpose(random.choices(classBT, k=int(len(classBT)*0.5)))
    
    # remove 20% from a subset of classA for which classA(1,:)<0 and 80% from a
    # subset of classA for which classA(1,:)>0
    # greater_than_0=[]
    # smaller_than_0=[]
    # for data_point in classAT:
    #     print(data_point)
    #     if data_point[1]>0:
    #         greater_than_0.append(data_point)
    #     else:
    #         smaller_than_0.append(data_point)
    
    # classA1 = random.choices(smaller_than_0, k=int(len(smaller_than_0)*0.8))
    # classA2 = random.choices(greater_than_0, k=int(len(greater_than_0)*0.2))
    # print(classA1)
    # print(classA2)

    # classA = np.transpose(np.concatenate((classA1, classA2), axis = 0))


    
    
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
    
    return patterns, targets, symmetric_targets, classA, classB

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

    legend = ['Error']
    plt.legend(legend, loc='upper left', fontsize=8)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title(title)
    #plt.savefig(title +".png")
    plt.savefig(title +".png")


# def plot_missclassification(epoch, missclassified):
#     plt.figure(4)
#     plt.plot([epoch],[missclassified], marker = "o")
#     plt.savefig("missclassified.png")


def plot_accuracy(y, title, nr, epoch, neg, pos):
    
    plt.figure(nr)
    
    plt.plot(y, marker='o', color='r', markersize=1, linewidth=0.8, label = "Accuracy")
    plt.plot(neg, label ="Accuracy for Negative class")
    plt.plot(pos, label = "Accuracy for Positive class")

   
    plt.legend(loc='upper left', fontsize=8)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    #plt.savefig(title +".png")
    plt.savefig("ANN_lab1/accuracy/"+"complicated" +".png")


def mse(errors, n):
    
    mse = 0
    
    for error in errors:
        mse += error**2
    mse = mse/n
    return mse   
    
    
def iterative_classification(patterns, targets, weights, type):

    
    patternsT = np.transpose(patterns)
    errors = [] 
    accuracy = []
    
    for epoch in range(EPOCHS):
        missclassified = 0
        change = 0
        epoch_error = []  
        weight_updates = [0,0,0]   
        
        
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
                title = "Sequential mode, Perceptron learning"
                if weighted_sum < 0:
                    current_class = 0
                else:
                    current_class = 1
                
                error = targets[pattern_index]-current_class
                
                if error != 0:
                    missclassified +=1
                
                # samma som delta 
                epoch_error.append(targets[pattern_index]-weighted_sum)
            
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
                title = "Sequential mode, Delta learning"
            
            for i in range(len(weights)):
                #print(LEARNING_RATE*error*patternsT[pattern_index][i])
                weights[i] += LEARNING_RATE*error*patternsT[pattern_index][i]
                change += LEARNING_RATE*error*patternsT[pattern_index][i]
                weight_updates[i] += LEARNING_RATE*error*patternsT[pattern_index][i]
        print("missclassified", missclassified)
        errors.append(np.linalg.norm(epoch_error))
        accuracy.append((len(targets)-missclassified)/len(targets))
        
        if np.linalg.norm(weight_updates) < CONVERGENCE:
            print(epoch," converged")
            break
                
    error_title = "Error per Epoch, Delta Learning, Learning Rate "+str(LEARNING_RATE)
    accuracy_title = "Accuracy per Epoch 50% Class A, 100% ClassB "

    print("Final Error: ", errors[-1])
    plot_error(np.linspace(0,EPOCHS,num=EPOCHS), errors, error_title, 7, epoch)
    plot_accuracy(accuracy, accuracy_title, 11112, epoch)
    plot_decision_boundary(weights, type, title, 11)
                
        # if type == "perceptron":
        #     plot_missclassification(epoch, missclassified)

def batch_classification(patterns, targets, weights):
    errors = []
    accuracy = []
    accuracy_pos = []
    accuracy_neg = []
    patternsT = np.transpose(patterns)

    for epoch in range(EPOCHS):
        error = targets-np.dot(weights, patterns[:])
        weight_updates = LEARNING_RATE*np.dot(error, patternsT[:])
        weights += weight_updates
        errors.append(np.linalg.norm(error))
        #print(np.linalg.norm(weight_updates))
        missclassified = 0
        pattern_index = 0
        true_neg = 0
        false_neg = 0
        true_pos = 0
        false_pos = 0
        
        for weighted_sum in np.dot(weights, patterns[:]):
            if weighted_sum < 0:
                current_class = -1
            else:
                current_class = 1
                
            if targets[pattern_index]==-1 and current_class == -1:
                true_neg+=1
            elif targets[pattern_index]==-1 and current_class == 1:
                false_pos+= 1
                missclassified +=1

            elif targets[pattern_index] == 1 and current_class ==1:
                true_pos +=1
            elif targets[pattern_index] == 1 and current_class == -1:
                false_neg +=1
                missclassified +=1
            pattern_index += 1
        
        #print("missclassified", missclassified)
        accuracy.append((true_neg+true_pos)/(true_neg+true_pos+false_neg+false_pos))
        accuracy_neg.append(true_neg/(true_neg+false_pos))
        accuracy_pos.append(true_pos/(true_pos+false_neg))
        
        #print(true_pos, true_neg, false_pos, false_neg)
       
        if np.linalg.norm(weight_updates) < CONVERGENCE:
            print(epoch," converged")
            break
        
    print("Final Error: ", errors[-1])
    error_title = "Error per Epoch in Batch Mode, Learning Rate "+str(LEARNING_RATE)
    accuracy_title = "Accuracy per Epoch, 80%"+" from classA(1,:)<0 and 20%"+" from classA(1,:)>0"
    plot_accuracy(accuracy, accuracy_title, 112, epoch, accuracy_neg, accuracy_pos)
    plot_error(np.linspace(0,EPOCHS,num=EPOCHS), errors, error_title, 8, epoch)
    plot_decision_boundary(weights, "delta", "Batch mode, Delta learning", 10)
            

def main():
    classA, classB = generate_data()
    plot_data(classA, classB, 1)
    patterns, targets, symmetric_targets, classA, classB = create_patterns_targets(classA, classB)
    # skicka in dimensions på vikter 
    weights = initialize_weights(patterns, targets)
    # plot_data(classA, classB, 11)
    # iterative_classification(patterns, targets, weights, type = "perceptron")
    # plot_data(classA, classB, 11)
    # iterative_classification(patterns, symmetric_targets, weights, type = "delta")
    plot_data(classA, classB, 10)
    batch_classification(patterns, symmetric_targets, weights)

if __name__ == '__main__':
    main()
    
 


