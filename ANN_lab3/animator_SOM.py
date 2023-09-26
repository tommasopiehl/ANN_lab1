import os

from matplotlib import animation, pyplot as plt, cm


class Animator():
    """
    Class to store matplotlib figures and save them as a gif
    """

    def __init__(self):
        self.figures = []  # list of figures provided by matplotlib in order
        self.frame_data = []  # list of data to be plotted in each frame
        self.current_figure = None  # index of the current figure
        self.current_ax = None  # index of the current axis

    def save_frame(self, data, centers):
        self.frame_data.append((data, centers))

    def animate(self, i):
        """
        Function to animate the figures
        """

        self.current_ax.clear()
        data, centers = self.frame_data[i]

        if data.shape[0] == 3:

            self.current_ax = self.current_figure.gca(projection='3d')
            self.current_ax.view_init(0, i)

            for i in range(centers.shape[0]):
                for j in range(centers.shape[1]):
                    self.current_ax.scatter(centers[i, j, 0], centers[i, j, 1], centers[i,j,2], color='red')


            # plot the lines
            for i in range(centers.shape[0]):
                for j in range(centers.shape[1]):
                    if i < centers.shape[0] - 1:
                        self.current_ax.plot([centers[i, j, 0], centers[i + 1, j, 0]],
                                             [centers[i, j, 1], centers[i + 1, j, 1]],
                                             [centers[i, j, 2], centers[i + 1, j, 2]], color='black')
                    if j < centers.shape[1] - 1:
                        self.current_ax.plot([centers[i, j, 0], centers[i, j + 1, 0]],
                                             [centers[i, j, 1], centers[i, j + 1, 1]],
                                             [centers[i, j, 2], centers[i, j + 1, 2]], color='black')

            # plot the data
            self.current_ax.scatter(data[0, :], data[1, :], data[2, :])

        elif data.shape[0] == 2:
            self.current_ax = self.current_figure.gca()
            # plot the centers
            for i in range(centers.shape[0]):
                for j in range(centers.shape[1]):
                    self.current_ax.scatter(centers[i, j, 0], centers[i, j, 1], color='red')

            # plot the lines
            for i in range(centers.shape[0]):
                for j in range(centers.shape[1]):
                    if i < centers.shape[0] - 1:
                        self.current_ax.plot([centers[i, j, 0], centers[i + 1, j, 0]],
                                 [centers[i, j, 1], centers[i + 1, j, 1]], color='black')
                    if j < centers.shape[1] - 1:
                        self.current_ax.plot([centers[i, j, 0], centers[i, j + 1, 0]],
                                 [centers[i, j, 1], centers[i, j + 1, 1]], color='black')

            # plot the data
            if data is not None:
                self.current_ax.scatter(data[0, :], data[1, :])


    def save(self, filename):
        """
        Adds each figure as a frame of a gif with 30 fps, and saves it to filename
        """
        plt.cla()
        plt.clf()
        self.current_figure = plt.figure()
        data = self.frame_data[0][0]
        if data.shape[0] == 2:
            self.current_ax = self.current_figure.gca()
        elif data.shape[0] == 3:
            self.current_ax = self.current_figure.gca(projection="3d")

        frames = len(self.frame_data)

        anim = animation.FuncAnimation(self.current_figure, self.animate, frames=frames, interval=20, blit=False)
        anim.save(filename, writer='imagemagick', fps=30)

    def save_png_sequence(self, filename):
        """
        Saves each frame as a png file in a sequence
        """

        # create a folder for the png files if it doesn't exist
        if not os.path.exists(filename):
            os.makedirs(filename)

        n_frames = len(self.frame_data)
        sample_rate = n_frames // 60



        for i_frame, frame in enumerate(self.frame_data):
            plt.cla()
            plt.clf()

            self.current_figure = plt.figure()
            data, centers = self.frame_data[i_frame]

            if i_frame % sample_rate == 0:

                if data.shape[0] == 3:
                    self.current_ax = self.current_figure.gca(projection='3d')
                    self.current_ax.view_init(0, i_frame)

                    # plot the centers
                    for i in range(centers.shape[0]):
                        for j in range(centers.shape[1]):
                            self.current_ax.scatter(centers[i, j, 0], centers[i, j, 1], centers[i, j, 2], color='red')

                    # plot the lines
                    for i in range(centers.shape[0]):
                        for j in range(centers.shape[1]):
                            if i < centers.shape[0] - 1:
                                self.current_ax.plot([centers[i, j, 0], centers[i + 1, j, 0]],
                                                     [centers[i, j, 1], centers[i + 1, j, 1]],
                                                     [centers[i, j, 2], centers[i + 1, j, 2]], color='black')
                            if j < centers.shape[1] - 1:
                                self.current_ax.plot([centers[i, j, 0], centers[i, j + 1, 0]],
                                                     [centers[i, j, 1], centers[i, j + 1, 1]],
                                                     [centers[i, j, 2], centers[i, j + 1, 2]], color='black')

                    # plot the data
                    self.current_ax.scatter(data[0, :], data[1, :], data[2, :])

                elif data.shape[0] == 2:
                    self.current_ax = self.current_figure.gca()
                    # plot the centers
                    for i in range(centers.shape[0]):
                        for j in range(centers.shape[1]):
                            self.current_ax.scatter(centers[i, j, 0], centers[i, j, 1], color='red')

                    # plot the lines
                    for i in range(centers.shape[0]):
                        for j in range(centers.shape[1]):
                            if i < centers.shape[0] - 1:
                                self.current_ax.plot([centers[i, j, 0], centers[i + 1, j, 0]],
                                                     [centers[i, j, 1], centers[i + 1, j, 1]], color='black')
                            if j < centers.shape[1] - 1:
                                self.current_ax.plot([centers[i, j, 0], centers[i, j + 1, 0]],
                                                     [centers[i, j, 1], centers[i, j + 1, 1]], color='black')

                    # plot the data
                    if data is not None:
                        self.current_ax.scatter(data[0, :], data[1, :])

                plt.savefig(filename + '/frame' + str(i_frame//sample_rate) + '.png')
                plt.close(self.current_figure)
