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


    def save_frame(self, xx, yy, z, data, classification):
        self.frame_data.append((xx, yy, z, data, classification))


    def animate(self, i):
        """
        Function to animate the figures
        """

        self.current_ax.clear()
        xx, yy, z, data, classification = self.frame_data[i]
        self.current_ax = self.current_figure.gca(projection='3d')

        self.current_ax.plot_surface(xx, yy, z*0.001, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        for i in range(len(data)):
            if classification[i] > 0:
                self.current_ax.scatter(data[i][0], data[i][1], 0.01, c='r', marker='o')
            else:
                self.current_ax.scatter(data[i][0], data[i][1], 0.01, c='b', marker='^')
        # View angle set to above the plane
        self.current_ax.view_init(90, 0)


    def save(self, filename):
        """
        Adds each figure as a frame of a gif with 30 fps, and saves it to filename
        """
        xx, yy, z, data, classification = self.frame_data[0]
        self.current_figure = plt.figure()
        self.current_ax = self.current_figure.gca(projection='3d')

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

        for i, frame in enumerate(self.frame_data):
            xx, yy, z, data, classification = frame

            # plot the heatmap
            plt.clf()
            plt.pcolormesh(xx, yy, z, cmap=cm.coolwarm)
            plt.colorbar()

            # plot the data points
            for j in range(len(data)):
                if classification[j] > 0:
                    plt.scatter(data[j][0], data[j][1], c='r', marker='o')
                else:
                    plt.scatter(data[j][0], data[j][1], c='b', marker='^')

            # save the figure
            plt.savefig(filename + '/frame' + str(i) + '.png')



