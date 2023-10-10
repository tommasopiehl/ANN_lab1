import os

from matplotlib import animation, pyplot as plt, cm


class Animator():
    """
    Class to store matplotlib figures and save them as a gif
    """

    def __init__(self, shape):
        self.shape = shape
        self.figures = []  # list of figures provided by matplotlib in order
        self.frame_data = []  # list of data to be plotted in each frame
        self.current_figure = None  # index of the current figure
        self.current_ax = None  # index of the current axis


    def save_frame(self, x):
        self.frame_data.append(x.copy())


    def animate(self, i):
        """
        Function to animate the figures
        """

        print(i)

        self.current_ax.clear()
        x = self.frame_data[i]
        print(x.shape)
        plt.imshow(x.reshape(self.shape), cmap='gray')
        plt.axis('off')

        self.current_ax = self.current_figure.gca()


    def save(self, filename):
        """
        Adds each figure as a frame of a gif with 30 fps, and saves it to filename
        """
        x = self.frame_data[0]
        self.current_figure = plt.figure()
        self.current_ax = self.current_figure.gca()

        frames = len(self.frame_data)

        anim = animation.FuncAnimation(self.current_figure, self.animate, frames=frames, interval=20, blit=False)
        anim.save(filename, writer='imagemagick', fps=160)

    def save_png_sequence(self, filename):
        """
        Saves each frame as a png file in a sequence
        """

        # create a folder for the png files if it doesn't exist
        if not os.path.exists(filename):
            os.makedirs(filename)

        self.current_figure = plt.figure()
        self.current_ax = self.current_figure.gca()


        for i, frame in enumerate(self.frame_data):
            if i % 32 == 0:
                self.current_figure = plt.figure()
                self.animate(i)

                # save the figure
                plt.savefig(filename + '/frame' + str(i//32) + '.png')
                plt.close(self.current_figure)



