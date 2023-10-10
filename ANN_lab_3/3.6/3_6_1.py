

import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLines

from ANN_lab_3.hopfield_net_bias import HopfieldNet_bias

if __name__ == '__main__':
    shape = (15, 15)
    size = shape[0] * shape[1]
    biases = [0.2, 0.15, 0.1, 0.05, 0.01]
    n_patterns = 500

    # create the hopfield network

    networks = []
    patterns = []
    patterns_learned = []
    n_recalled_list = []

    for bias in biases:
        networks.append(HopfieldNet_bias(size, shape, sequential=False))

        # create array with bias*100% of 1s and (1-bias)*100% of 0s
        base = np.zeros(size)
        base[:int(size * bias)] = 1

        patterns.append(np.zeros((n_patterns, size)))
        for i in range(n_patterns):
            patterns[-1][i] = base.copy()
            np.random.shuffle(base)

        n_recalled_list.append([])
        patterns_learned.append(None)

        # count number of unique patterns
        unique_patterns = []

        for pattern in patterns[-1]:
            match = False
            for unique_pattern in unique_patterns:
                if np.array_equal(pattern, unique_pattern) or np.array_equal(pattern, -unique_pattern):
                    match = True
                    break
            if not match:
                unique_patterns.append(pattern)

        print(f"With bias {bias}, there are {len(unique_patterns)} unique patterns")


    for i in range(n_patterns):
        for j in range(len(biases)):
            if patterns_learned[j] is None:
                patterns_learned[j] = patterns[j][i].copy().reshape(1, -1)
            else:
                patterns_learned[j] = np.vstack((patterns_learned[j], patterns[j][i].copy()))

            networks[j].train(patterns_learned[j], bias=True)

            n_recalled = 0
            for pattern in patterns_learned[j]:
                output = networks[j].run(pattern.copy(), sequential=True)
                if np.array_equal(output, pattern) or np.array_equal(output, -pattern):
                    n_recalled += 1
            n_recalled_list[j].append(n_recalled)

        print(f"Storing {i+1} patterns")


    fig = plt.figure(figsize=(8, 3))
    # draw horizontal line at theoretical capacity
    theoretical_capacity = 0.138 * size
    plt.axhline(theoretical_capacity, color="red", linestyle="--", label="Theoretical capacity", xmax=n_patterns-1)
    labelLines(plt.gca().get_lines(), xvals=[0.75*(n_patterns-1)], ha="left", zorder=2.5)

    for i in range(len(biases)):
        plt.plot(n_recalled_list[i], label=f"{round(biases[i]*100, 1)}% - {round(100-biases[i]*100, 1)}% split")

    plt.legend(fontsize=8)
    plt.xlabel("Number of patterns stored")
    plt.ylabel("Number of patterns recalled")
    plt.tight_layout()

    plt.show()
