import itertools

import numpy as np
from matplotlib import pyplot as plt

from ANN_lab_3.hopfield_net import HopfieldNet, find_two_largest_factors


def display_image(pict, shape, title=None, show=True, fig=None, ax=None):
    """
    Display an image.
    :param pict:
    :param shape:
    :param title:
    :param show:
    :param fig:
    :param ax:
    :return:
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.imshow(pict.reshape(shape), cmap='gray')
    ax.axis('off')
    if title is not None:
        ax.set_title(title)
    if show:
        plt.show()
    else:
        return fig, ax


if __name__ == '__main__':
    pict_shape = (32, 32)
    pict_size = pict_shape[0] * pict_shape[1]
    s = 0.1  # noise level

    # import data from .dat file
    train_data = np.loadtxt("data/pict.dat", delimiter=",", dtype=int).reshape(11, pict_size)

    # Display the patterns
    fig, axs = plt.subplots(3, 4)

    for i in range(11):
        display_image(train_data[i], pict_shape, title=f"Pattern {i + 1}", show=False, fig=fig, ax=axs[i // 4, i % 4])

    axs[2, 3].axis('off')
    plt.show()
    plt.close(fig)

    # Find all sets of patterns in size 11 to 3
    range_11 = np.arange(11)

    set_list = []

    for i in range(1, 12):
        set_list.append(np.array(list(itertools.combinations(range_11, i))))

    hopfield = HopfieldNet(pict_size, pict_shape)

    for set in set_list:
        test_n = len(set[0])
        print(f"Testing set of size {test_n}")

        n_stable = 0

        for comb in set:

            test = np.zeros((test_n, pict_size))
            for i in range(test_n):
                test[i] = train_data[comb[i]]

            hopfield.train(test, sequential=True)

            # Test the network
            # fig, axs = plt.subplots(3, test_n)

            # make plot square
            # fig.set_size_inches(8 * (test_n / 3), 8)

            # bottom row is three first patterns
            show_gif = False

            all_stable = True

            for i, data in enumerate(test):
                input = data.copy()
                # display_image(input, pict_shape, title=f"Pattern {i}", show=False, fig=fig, ax=axs[2, i])
                input_noisy = input.copy()
                input_noisy[np.random.randint(0, pict_size, int(s * pict_size))] *= -1
                # display_image(input_noisy, pict_shape, title=f"input {i} with {s * 100}% noise", show=False, fig=fig,
                #              ax=axs[0, i])
                output = hopfield.run(input_noisy, sequential=True, show_gif=show_gif, update_order="random",
                                      gif_name="p10")
                # display_image(output, pict_shape, title=f"output p10", show=False, fig=fig, ax=axs[1, i])

                if not (np.array_equal(output, input) or np.array_equal(output, -input)):
                    all_stable = False

            if all_stable:
                n_stable += 1
            # plt.show()

        print(f"Number of stable sets: {n_stable} / {len(set)}")
