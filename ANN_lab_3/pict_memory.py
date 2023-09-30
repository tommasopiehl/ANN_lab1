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
    test_n = 3

    # import data from .dat file
    train_data = np.loadtxt("data/pict.dat", delimiter=",", dtype=int).reshape(11, pict_size)

    # Display the patterns
    fig, axs = plt.subplots(3,4)

    for i in range(11):
        display_image(train_data[i], pict_shape, title=f"Pattern {i+1}", show=False, fig=fig, ax=axs[i//4, i%4])

    axs[2, 3].axis('off')
    plt.show()
    plt.close(fig)


    hopfield = HopfieldNet(pict_size, pict_shape)

    test = np.zeros((test_n, pict_size))
    for i in range(test_n):
        test[i] = train_data[i]

    hopfield.train(test, sequential=True)

    # Test the network
    test_dim_1, test_dim_2 = find_two_largest_factors(test_n)

    one_dim = False
    if test_dim_1 == 1:
        one_dim = True

    fig, axs = plt.subplots(test_dim_1, test_dim_2)

    for i in range(test_n):
        output = hopfield.run(train_data[i], sequential=True, update_order="random")

        if one_dim:
            display_image(output, pict_shape, title=f"Pattern {i}", show=False, fig=fig, ax=axs[i])
        else:
            display_image(output, pict_shape, title=f"Pattern {i}", show=False, fig=fig, ax=axs[i//test_dim_1, i%test_dim_2])

    if not one_dim:
        for i in range(test_dim_1):
            for j in range(test_dim_2):
                axs[i, j].axis('off')

    plt.show()
    plt.close(fig)

    # Test the network
    fig, axs = plt.subplots(3, 3)

    # make plot square
    fig.set_size_inches(8, 8)

    # bottom row is three first patterns
    for i in range(3):
        output = hopfield.run(train_data[i], sequential=True, update_order="random")
        display_image(output, pict_shape, title=f"Pattern {i}", show=False, fig=fig, ax=axs[2, i])

    # middle row is pattern 10, 11 and 1 with 10% noise
    pattern_10 = train_data[9]
    pattern_11 = train_data[10]
    pattern_1 = train_data[0]

    # add noise to pattern 1
    pattern_1_noisy = pattern_1.copy()
    pattern_1_noisy[np.random.randint(0, pict_size, int(0.1*pict_size))] *= -1

    # top row is pattern 10, 11 and 1 with 30% noise
    display_image(pattern_10, pict_shape, title=f"Pattern 10", show=False, fig=fig, ax=axs[0, 0])
    display_image(pattern_11, pict_shape, title=f"Pattern 11", show=False, fig=fig, ax=axs[0, 1])
    display_image(pattern_1_noisy, pict_shape, title=f"Pattern 1 with 10% noise", show=False, fig=fig, ax=axs[0, 2])

    # show results
    output_10 = hopfield.run(pattern_10, sequential=True, update_order="random")
    output_11 = hopfield.run(pattern_11, sequential=True, update_order="random")
    output_1 = hopfield.run(pattern_1_noisy, sequential=True, update_order="random")

    display_image(output_10, pict_shape, title=f"Pattern 10", show=False, fig=fig, ax=axs[1, 0])
    display_image(output_11, pict_shape, title=f"Pattern 11", show=False, fig=fig, ax=axs[1, 1])
    display_image(output_1, pict_shape, title=f"Pattern 1 with 10% noise", show=False, fig=fig, ax=axs[1, 2])



    plt.show()






