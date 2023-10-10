import numpy as np
import matplotlib.pyplot as plt
from ANN_lab_3.hopfield_net import HopfieldNet

shape = (10, 10)
size = shape[0] * shape[1]

theoretical_capacity = 0.138 * size
n_grids = int(theoretical_capacity * 3)
n_tests = 100

print(f"Theoretical capacity: {theoretical_capacity}")

n_patterns = []
percentages = []
recall_avg = []
recall_std = []

for i in range(1, n_grids):
    print(f"Storing {i} patterns")
    n_correct = 0
    recalls = []

    for j in range(n_tests):
        recalled = 0
        patterns = np.random.choice([-1, 1], size=(i, size))

        hopfield = HopfieldNet(size, shape=shape)
        hopfield.train(patterns, sequential=True)

        all_good = True

        for pattern in patterns:
            output = hopfield.run(pattern.copy(), sequential=True)
            if not (np.array_equal(output, pattern) or np.array_equal(output, -pattern)):
                all_good = False
            else:
                recalled += 1

        if all_good:
            n_correct += 1
        recalls.append(recalled / i)

    recall_avg.append(np.mean(recalls))
    recall_std.append(np.std(recalls))
    n_patterns.append(i)
    percentages.append(n_correct * 100 / n_tests)

fig = plt.figure(figsize=(8, 3))
plt.title("Capacity of Hopfield network")

plt.plot(n_patterns, percentages, label="Networks with perfect recall")

# Plot recall average and std
recall_avg = np.array(recall_avg)
recall_std = np.array(recall_std)
n_patterns = np.array(n_patterns)

plt.plot(n_patterns, 100 * np.maximum(0, recall_avg - recall_std), label="Standard deviation from average recall",
         color="orange", linestyle="--", alpha=0.5)
plt.plot(n_patterns, 100 * np.minimum(1, recall_avg + recall_std), color="orange", linestyle="--", alpha=0.5)
plt.plot(n_patterns, 100 * recall_avg, label="average recall", color="green")

plt.axvline(theoretical_capacity, color="red", label="Theoretical capacity")

plt.xlabel("Number of patterns")
plt.ylabel("Percentage (%)")
plt.legend(fontsize=8)

plt.tight_layout()

plt.show()
