import numpy as np

LEARNING_RATE = 0.1
targets = np.array([1,2,3])
patterns = np.array([[1,2,3], [1,2,3], [1,1,1]])
patternsT = np.transpose(patterns)
print(patternsT)
weights = [0.1, 0.2, 0.3]
weightsT = np.transpose(weights)


for i in range(10):
    error = targets-np.dot(weightsT,patterns[:])
    print("error", error)
    weight_updates = LEARNING_RATE*np.dot(error, patterns[:])

    weights += weight_updates

    print(weight_updates)
    print(weights)
    weightsT = np.transpose(weights)
