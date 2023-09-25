import random
import numpy as np

ratio = 0.5
patterns = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])
targets = np.array([[-1,-2,-3,-4,-5,-6]])

nsamp = int(len(patterns[0])*ratio)

perm_indices = np.random.permutation(patterns.shape[1])
# Select the first 'nsamp' indices for training
training_indices = perm_indices[:nsamp]

# Use these indices to select the corresponding patterns and targets
training_patterns = patterns[:, training_indices]
training_targets = targets[:, training_indices]

print(training_patterns)
print(training_targets)
patterns = np.transpose(patterns)
training_patterns = np.transpose(training_patterns)