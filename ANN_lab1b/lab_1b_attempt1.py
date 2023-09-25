import numpy as np
import random
from graphics import *
from create_data_patterns_targets import *
import math 
from error_computations import *
from matplotlib.pyplot import cm

np.random.seed(42)
LEARNING_RATE = 0.001
EPOCHS = 1000
ALPHA = 1.5

def transfer_func(x):
    return (2/(1+np.exp(-x)))-1


def transfer_func_derivative(x):
    
    # här kan man ist skicka in transfer_func värdet direkt 
    # return 2*math.exp(x)/pow(math.exp(x)+1, 2)   
    return (1+transfer_func(x))(1-transfer_func(x))/2

def forward_pass(patterns, w, v):
    
    hin = np.dot(w,np.vstack((np.transpose(patterns), np.ones((1,len(patterns))))))
    hout = transfer_func(hin)
    
    # add bias again 
    hout = np.vstack((hout, np.ones((1,len(patterns)))))
    oin = np.dot(v, hout)
    
    out = transfer_func(oin)
    
    return out, hout

def backward_pass(targets, w, v, out, hout, hidden_nodes):
    
    # error signal computed for each node
    # must start in output layer and propagate 
    # layer by layer 
    
    delta_o = (out-targets) * ((1+out)*(1-out))*0.5
    delta_h = np.dot(np.transpose(v), delta_o)*((1+hout)*(1-hout))*0.5
    delta_h = delta_h[:hidden_nodes,:]
    return delta_o, delta_h
    
    
def weight_update(patterns, w, v, delta_o, delta_h, hout, dw, dv):
    # patterns with bias
    pat = np.vstack((np.transpose(patterns), np.ones((1,len(patterns)))))
    dw = (dw* ALPHA ) - np.dot(delta_h, np.transpose(pat))*(1-ALPHA)
    dv = (dv * ALPHA ) - np.dot(delta_o, np.transpose(hout))*(1-ALPHA)
    
    w = w + dw*LEARNING_RATE
    v = v + dv *LEARNING_RATE
    
    return w, v, dw, dv

def MLP_batch_function_approx(patterns, targets, hidden_nodes, w, v, dw, dv):
    for epoch in range(EPOCHS):
        out, hout = forward_pass(patterns, w, v)
        delta_o, delta_h = backward_pass(targets, w, v, out, hout, hidden_nodes)
        w, v, dw, dv = weight_update(patterns, w, v, delta_o, delta_h, hout, dw, dv) 
    return out
    

def MLP_batch(patterns, targets, w,v, dw, dv, hidden_nodes, c, testing_patterns, testing_targets, classA_testing, classB_testing):
    
    mse_training = []
    missclassified_training = []
    
    mse_testing = []
    missclassified_testing = []
    
    for epoch in range(EPOCHS):
        out, hout = forward_pass(patterns, w, v)
        
        delta_o, delta_h = backward_pass(targets, w, v, out, hout, hidden_nodes)
        # delta_o = felet i output-layer
        mse_training.append(compute_mse(delta_o))
        missclassified_training.append(compute_missclassified(out, targets))
        
        mse_testing.append(compute_mse_testing(w, v, testing_patterns, testing_targets, hidden_nodes))
        missclassified_testing.append(compute_missclassified_testing(testing_patterns, testing_targets,w,v))
        
        # if(missclassified_training[-1] == 0):
        #     print(epoch)
        w, v, dw, dv = weight_update(patterns, w, v, delta_o, delta_h, hout, dw, dv) 
    
    # Decision Boundary Training
    plot_decision_boundary(w, v, "Decision boundary, training data", 101, patterns, targets)
    
    # Decision Boundary Testing
    plot_decision_boundary(w, v, "Decision boundary, testing data", 102, testing_patterns, testing_targets)
    
    # Training 
    label = str(hidden_nodes) +" hidden nodes"
    plot_error(mse_training, "MSE for the training data", 99, c, label)
    plot_error(missclassified_training, "Missclassified for the training data", 100, c, label)
    print("Missclassified in training data", missclassified_training[-1])
    
    # Testing
    label = str(hidden_nodes) +" hidden nodes"
    plot_error(mse_testing, "MSE for the testing data", 115, c, label)
    plot_error(missclassified_testing, "Missclassified for the testing data", 116, c, label)
    print("Missclassified in testing data", missclassified_testing[-1])
    
def function_approximation():
    
    # Define the range and step for x and y
    x = np.arange(-5, 5.5, 0.5)
    y = np.arange(-5, 5.5, 0.5)

    # Create a meshgrid for x and y
    [xx, yy] = np.meshgrid(x, y)

    # Calculate z based on the given formula
    z = np.exp(-xx**2 * 0.1) * np.exp(-yy**2 * 0.1) - 0.5

    # Create a mesh plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, z)

    # Set labels and title
    # Set axis limits
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-0.7, 0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Function to be approximated')

    # Show the plot
    plt.savefig("3Dplot.png")

    # Reshape z into targets
    ndata = len(x) * len(y)
    targets = np.array(z.reshape(1, ndata))

    ratios = [0.7]
    color = iter(cm.rainbow(np.linspace(0, 1, len(ratios))))
    xx, yy = np.meshgrid(x, y)

    for ratio in ratios:
        # Create patterns
        patterns = np.array([xx.reshape(1, ndata)[0], yy.reshape(1,ndata)[0]])

        
        nsamp = int(len(patterns[0])*ratio)

        # denna ska int ha punker som rader

        perm_indices = np.random.permutation(patterns.shape[1])
       
        # Select the first 'nsamp' indices for training
        training_indices = perm_indices[:nsamp]

        # Use these indices to select the corresponding patterns and targets
        training_patterns = patterns[:, training_indices]
        training_targets = targets[:, training_indices]
        patterns = np.transpose(patterns)
        training_patterns = np.transpose(training_patterns)
    


    # ner hit är det nog rätt



        # Train the network 
        #hidden_nodes = [1,3,5,10,15,20,23,25]
        hidden_nodes = [10]

        for hidden_node in hidden_nodes:
            w = initialize_W(training_patterns, hidden_node)
            v = initialize_V(training_targets, hidden_node)
            dw, dv = initialize_dw_dv(hidden_node, training_patterns, training_targets)
            
            
            mse_training = []
            mse_testing = []
            
            last_mse = 10
            
            for epoch in range(EPOCHS):
                out, hout = forward_pass(training_patterns, w, v)
                delta_o, delta_h = backward_pass(training_targets, w, v, out, hout, hidden_node)
                mse_training.append(compute_mse(delta_o))
                mse_testing.append(compute_mse_testing(w, v,patterns, targets, hidden_node))
                w, v, dw, dv = weight_update(training_patterns, w, v, delta_o, delta_h, hout, dw, dv) 

                if (mse_training[-1])<0.001:
                    print(epoch)
                    break
                last_mse = mse_training[-1]
            
            #out = MLP_batch_function_approx(patterns, targets, hidden_node, w, v, dw, dv)
            # Reshape out into zz for another mesh plot
            
            
            gridsize = len(x)
            out, hout = forward_pass(patterns, w, v)
            zz = out.reshape(gridsize, gridsize)

            # Create another mesh plot
            fig = plt.figure(1111)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(xx, yy, zz)

            # Set axis limits
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_zlim(-0.7, 0.7)

            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.title("Approximated function 80%"+" used for training")

            # Show the plot
            plt.savefig("out.png")
            # Training 
            print("hidden nodes = " , hidden_node)
            print("ratio= ", ratio )
            #label = str(hidden_node) +" hidden nodes"
            label = str(ratio*100)+"%"+" used for training"
            c = next(color)
            plot_error(mse_training,"MSE for the training data", 99, c, label)
            #plot_error(missclassified_training, "Missclassified for the training data", 100, "r", label)
            #print("Missclassified in training data", missclassified_training[-1])
            print("MSE in training data", mse_training[-1])
            # Testing
            #label = str(hidden_node) +" hidden nodes"
            plot_error(mse_testing, "MSE for the testing data", 115, c, label)
            print("MSE in testing data", mse_testing[-1])
            print("difference ", mse_testing[-1]/mse_training[-1])
            #plot_error(missclassified_testing, "Missclassified for the testing data", 116, "r", label)
            #print("Missclassified in testing data", missclassified_testing[-1])
                

def main():
    # classA, classB = generate_data(100, 0.5,[1.0, 0.5], 0.5, [-1.5, -1.5])
    classA, classB = generate_non_linearly_seperable_data()
    patterns, targets, classA, classB, testing_patterns, testing_targets, classA_testing, classB_testing = create_patterns_targets(classA, classB)
    
    hidden_nodes_list = [1,2,3,4,5,6,10,15,19,20,21,22,23,24,25]
    color = iter(cm.rainbow(np.linspace(0, 1, len(hidden_nodes_list))))

    # for hidden_nodes in hidden_nodes_list:
        
    #     w = initialize_W(patterns, hidden_nodes)
    #     v = initialize_V(targets, hidden_nodes)
    #     dw, dv = initialize_dw_dv(hidden_nodes, patterns, targets)
    #     MLP_batch(patterns, targets, w, v, dw, dv, hidden_nodes, next(color), testing_patterns, testing_targets, classA_testing, classB_testing)
    
    # plot_data(classA, classB, 1, "ANN_lab1b/images/data_distribution.png")
    function_approximation()
    

if __name__ == '__main__':
    main()