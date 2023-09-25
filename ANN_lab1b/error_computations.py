
import numpy as np

# compute MSE
np.random.seed(42)

def transfer_func(x):
    return (2/(1+np.exp(-x)))-1

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
    
    

def compute_mse_testing(w, v, testing_patterns, testing_targets, hidden_nodes):
    mse = 0
    
    out, hout = forward_pass(testing_patterns, w, v) 
    delta_o, delta_h = backward_pass(testing_targets, w, v, out, hout, hidden_nodes)
    
    for error in delta_o[0]:
        mse += error**2
    mse = mse/len(delta_o[0])
    return mse 

def compute_missclassified_testing(testing_patterns, testing_targets, w, v):
    missclassified = 0
    out, hout = forward_pass(testing_patterns, w, v) 
    for target in range(len(testing_targets[0])):
        if out[0][target] < 0 and testing_targets[0][target] == 1:
            missclassified += 1
            
        elif out[0][target]>=0 and testing_targets[0][target] == -1:
            missclassified += 1
    
    return missclassified/len(testing_targets[0])

    

def compute_mse(delta_o):
   
    mse = 0
    for error in delta_o[0]:
        mse += error**2
    mse = mse/len(delta_o[0])
    return mse 

def compute_missclassified(out, targets):
    
    missclassified = 0
    
    for target in range(len(targets[0])):
        if out[0][target] < 0 and targets[0][target] == 1:
            missclassified += 1
            
        elif out[0][target]>=0 and targets[0][target] == -1:
            missclassified += 1
    
    return missclassified/len(targets[0])
            
      
    