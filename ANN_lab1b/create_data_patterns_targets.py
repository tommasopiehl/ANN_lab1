import numpy as np
import random

np.random.seed(42)


def generate_data(n, sigmaA, mA, sigmaB, mB):
    
    # sigma = standard deviation
    # m = medelvärde
    
    classA = np.zeros((2, n))
    classB = np.zeros((2, n))
    
    classA[0, :] = np.random.randn(1, n) * sigmaA + mA[0]
    classA[1, :] = np.random.randn(1, n) * sigmaA + mA[1]
    classB[0, :] = np.random.randn(1, n) * sigmaB + mB[0]
    classB[1, :] = np.random.randn(1, n) * sigmaB + mB[1]
    return classA, classB 

def generate_non_linearly_seperable_data():
    ndata = 100
    mA = [1.0, 0.3]
    sigmaA = 0.2
    mB = [0.0, -0.1]
    sigmaB = 0.3

    classA = np.zeros((2, ndata))
    classA[0, :] = np.concatenate((np.random.randn(1, round(0.5 * ndata)) * sigmaA - mA[0],np.random.randn(1, round(0.5 * ndata)) * sigmaA + mA[0]), axis=1) 
    classA[1, :] = np.random.randn(1, round(ndata)) * sigmaA + mA[1]

    classB = np.zeros((2, ndata))
    classB[0, :] = np.random.randn(1, ndata) * sigmaB + mB[0]
    classB[1, :] = np.random.randn(1, ndata) * sigmaB + mB[1]
    return classA, classB 
    
    
def create_patterns_targets(classA, classB):
    
    classAT = np.transpose(classA)
    np.random.shuffle(classAT)
    classBT = np.transpose(classB)
    np.random.shuffle(classBT)
    
    
    # 25% from each class used for training     
    classA_testing = np.transpose(classAT[0:int(len(classAT)*0.25)])
    classA = np.transpose(classAT[int(len(classAT)*0.25):len(classAT)])
    
    classB_testing = np.transpose(classBT[0:int(len(classBT)*0.25)])
    classB = np.transpose(classBT[int(len(classBT)*0.25):len(classBT)])
    
    # 50% from classA
    
    # classA_testing = np.transpose(classAT[0:int(len(classAT)*0.5)])
    # classA = np.transpose(classAT[int(len(classAT)*0.5):len(classAT)])
    
    # classB_testing = None
    # classB = np.transpose(classBT)
    
    
    # remove 50 from B
    # classB = np.transpose(random.choices(classBT, k=int(len(classBT)*0.5)))
    
    # remove 20% from a subset of classA for which classA(1,:)<0 and 80% from a
    # subset of classA for which classA(1,:)>0
    
    # greater_than_0=[]
    # smaller_than_0=[]
    # for data_point in classAT:
    #     if data_point[0]>0:
    #         greater_than_0.append(data_point)
    #     else:
    #         smaller_than_0.append(data_point)
    # smaller_than_0 = np.array(smaller_than_0)
    # greater_than_0 = np.array(greater_than_0)
    
    # classA1_testing = smaller_than_0[0: int(len(smaller_than_0)*0.2)]
    # classA1_training = smaller_than_0[int(len(smaller_than_0)*0.2): len(smaller_than_0)]
    # classA2_testing = greater_than_0[0:int(len(greater_than_0)*0.8)]
    # classA2_training = greater_than_0[int(len(greater_than_0)*0.8):len(greater_than_0)]
    
    
    
    # classA = np.transpose(np.concatenate((classA1_training, classA2_training), axis = 0))
    # classA_testing = np.transpose(np.concatenate((classA1_testing, classA2_testing), axis = 0))


    # classB_testing = None
    # classB = np.transpose(classBT)
    
    # targets_classA = np.ones(len(classA[0]))
    # targets_classB = np.zeros(len(classB[0]))
    # targets = np.append(targets_classA, targets_classB)
    
    
    # Spara
    
    targets_classA = np.ones([len(classA[0])])
    targets_classB = np.full([len(classB[0])],-1)
    targets = np.array([np.concatenate([targets_classA, targets_classB])])
    
    # skabort
    # targets = np.array([np.ones([len(classA[0])])])

    ## combining matrixes
    
    testing_targets_classA = np.ones(len(classA_testing[0]))
    testing_targets_classB = np.full([len(classB_testing[0])],-1)
    testing_targets = np.array([np.concatenate([testing_targets_classA, testing_targets_classB])])    
    
    patterns = np.vstack((np.transpose(classA), np.transpose(classB)))
    #patterns = np.transpose(classA)
    
    testing_patterns  = np.vstack((np.transpose(classA_testing), np.transpose(classB_testing)))
    #testing_patterns = np.transpose(classA_testing)
    ## adding a row of ones at the end
    # patterns = np.concatenate((patterns, [np.ones(len(patterns[0]))]), axis = 0)
    
    # shuffle 
    # shuffle = np.random.permutation(len(targets))
    # patterns = patterns[:, shuffle]
    # targets = targets[shuffle]
    
    return patterns, targets, classA, classB, testing_patterns, testing_targets, classA_testing, classB_testing

def initialize_V(targets, NR_NODES_HL):  
    v = np.zeros((len(targets), NR_NODES_HL+1))
    sigma = 0.5
    mean = 0
    
    for row in range(len(v)):
        v[row, :] = np.random.randn(1, NR_NODES_HL+1) * sigma + mean
            
    return v
    
def initialize_W(patterns, NR_NODES_HL):
    # patterns har redan fått en extra dim
    dim = len(patterns[0])+1
    w = np.zeros((NR_NODES_HL, dim))
    sigma = 0.5
    mean = 0
    
    for row in range(len(w)):
        w[row, :] = np.random.randn(1, dim) * sigma + mean
    return w

def initialize_dw_dv(NR_NODES_HL, patterns, targets):
    dw = np.zeros((NR_NODES_HL, len(patterns[0])+1))
    dv = np.zeros((len(targets), NR_NODES_HL+1))
    
    return dw, dv