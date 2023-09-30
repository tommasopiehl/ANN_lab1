import numpy as np

from ANN_lab_3.animator import Animator


def find_two_largest_factors(n):
    factor1 = 1

    # Find the first factor that divides n
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            factor1 = i
            break

    # Calculate the second factor
    factor2 = n // factor1

    return factor1, factor2


class HopfieldNet:
    """
    Hopfield network class.
    """
    def __init__(self, n_nodes, shape=None):
        self.weights = None
        self.threshold = None
        self.n_nodes = n_nodes
        self.update_order = np.arange(n_nodes)
        if shape is None:
            self.shape = find_two_largest_factors(n_nodes)
        else:
            self.shape = shape

    def train(self, data, sequential=False):
        """
        Train the network on the given data.
        :param data: The data to train on.
        """
        # Initialize the weights and threshold
        self.weights = np.zeros((self.n_nodes, self.n_nodes))
        self.threshold = np.zeros(self.n_nodes)

        # Calculate the weights

        for x in data:
            self.weights += np.outer(x, x)

        self.weights /= self.n_nodes
        if sequential:
            # clear diagonal
            np.fill_diagonal(self.weights, 0)


    def run_cycle(self, x, sequential=False, update_order=None):
        """
        Run one cycle of the network.
        :param x: The input to the network.
        :return: The output of the network.
        """
        if sequential:
            if update_order is None:
                pass
            elif update_order == "random_random":
                self.update_order = np.random.permutation(self.n_nodes)

            for node in self.update_order:
                x[node] = self.activation(x, node)

            return x
        else:
            return np.sign(np.dot(self.weights, x) - self.threshold)

    def run(self, x, n_cycles=100, sequential=False, show_gif=False, update_order=None):
        """
        Run the network for a number of cycles.
        :param x: The input to the network.
        :param n_cycles: The number of cycles to run.
        :param sequential: Whether to run the network sequentially or not.
        :param show_gif: Whether to show a gif of the network running.
        :param update_order: The order to update the nodes in. If None, the order is the order of the nodes. If
                            "random", the order is random. If "random_random", the order is random for each cycle.
        :return: The output of the network.
        """
        if show_gif:
            anim = Animator(self.shape)
            anim.save_frame(x)

        if sequential:
            if update_order is None:
                pass
            elif "random" in update_order:
                self.update_order = np.random.permutation(self.n_nodes)

        for i in range(n_cycles):
            x_old = x.copy()
            if sequential:
                self.run_cycle(x, sequential=True, update_order=update_order)
            else:
                x = self.run_cycle(x)
                if show_gif:
                    anim.save_frame(x)
            if np.array_equal(x, x_old):
                break
        if show_gif:
            anim.save(f"images/hopfield_{len(x)}.gif")
        return x

    def activation(self, x, i):
        """
        Calculate the activation of a node.
        :param x: The input to the network.
        :param i: The index of the node.
        :return: The activation of the node.
        """
        return np.sign(np.dot(self.weights[i], x) - self.threshold[i])


if __name__ == '__main__':
    x1d = np.array([1, -1, 1, -1,
                    1, -1, -1, 1])
    x2d = np.array([1, 1, -1, -1,
                    -1, 1, -1, -1])
    x3d = np.array([1, 1, 1, -1,
                    1, 1, -1, 1])

    x = np.array([x1d, x2d, x3d])

    hopfield = HopfieldNet(8)

    hopfield.train(x)

    # Iterate over all possible inputs
    output = []
    for i in range(2 ** 8):
        x_in = np.array([1 if digit == '1' else -1 for digit in f"{i:08b}"])
        out = hopfield.run(x_in)
        if np.array_equal(out, x_in):
            output.append(out)

    print("Number of attractors:", len(output))
    print("Attractors:")
    for attractor in output:
        if np.array_equal(attractor, x1d):
            print(f"{attractor} (= x1d)")
        elif np.array_equal(attractor, x2d):
            print(f"{attractor} (= x2d)")
        elif np.array_equal(attractor, x3d):
            print(f"{attractor} (= x3d)")
        elif np.array_equal(attractor, -x1d):
            print(f"{attractor} (= -x1d)")
        elif np.array_equal(attractor, -x2d):
            print(f"{attractor} (= -x2d)")
        elif np.array_equal(attractor, -x3d):
            print(f"{attractor} (= -x3d)")
        else:
            print(attractor)

    # Run the network on a noisy input
    x3d_noisy = np.array([-1, 1, 1, -1,
                          -1, 1, -1, 1])
    out = hopfield.run(x3d_noisy, show_gif=True)

    print(hopfield.weights)





