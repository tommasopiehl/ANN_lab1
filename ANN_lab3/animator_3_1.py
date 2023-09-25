import os

import numpy as np
from matplotlib import animation, pyplot as plt, cm


class Animator_3_1():
    """
    Class to store matplotlib figures and save them as a gif
    """

    def __init__(self):
        self.figures = []  # list of figures provided by matplotlib in order
        self.frame_data = []  # list of data to be plotted in each frame
        self.current_figure = None  # index of the current figure
        self.current_ax = None  # index of the current axis

    def save_frame(self, x, y1, y2, title, xlabel, ylabel, legend1, legend2):
        self.frame_data.append((x, y1, y2, title, xlabel, ylabel, legend1, legend2))

    def animate(self, i):
        """
        Function to animate the figures
        """

        self.current_ax.clear()
        x, y1, y2, title, xlabel, ylabel, legend1, legend2 = self.frame_data[i]
        self.current_ax = self.current_figure.gca()
        y2 = y2.reshape(-1, 1)
        self.current_ax.plot(x, y1, label=legend1)
        self.current_ax.plot(x, y2, label=legend2)
        self.current_ax.set_title(title)
        self.current_ax.set_xlabel(xlabel)
        self.current_ax.set_ylabel(ylabel)
        self.current_ax.set_ylim([-1.3, 1.3])
        self.current_ax.set_xlim([0, 2 * np.pi])
        self.current_ax.legend()

    def save(self, filename):
        """
        Adds each figure as a frame of a gif with 30 fps, and saves it to filename
        """
        self.current_figure = plt.figure()
        self.current_ax = self.current_figure.gca()

        frames = len(self.frame_data)
        anim = animation.FuncAnimation(self.current_figure, self.animate, frames=frames, interval=20, blit=False)
        anim.save(filename+".gif", writer='Pillow', fps=6)

    def save_png_sequence(self, filename):
        """
        Saves each frame as a png file in a sequence
        """

        # create a folder for the png files if it doesn't exist
        if not os.path.exists(filename):
            os.makedirs(filename)

        for i, frame in enumerate(self.frame_data):
            x, y1, y2, title, xlabel, ylabel, legend1, legend2 = frame

            fig, ax = plt.subplots()
            ax.plot(x, y1, label=legend1)
            ax.plot(x, y2, label=legend2)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend()

            # set axis limits
            ax.set_ylim([-1.3, 1.3])
            ax.set_xlim([0, 2*np.pi])

            # save the figure
            plt.savefig(filename + '/frame' + str(i) + '.png')
            plt.close()
