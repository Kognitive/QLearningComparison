import numpy as np
import matplotlib.pyplot as plt
from os import makedirs, path
from collection.ColorCollection import ColorCollection

class DirectoryManager:

    def __init__(self, directory, name, agent):

        # check if path exists
        joined = path.join(directory, name, agent)
        if not path.exists(joined):
            makedirs(joined)

        self.root = joined

    def write_policy_params(self, policies):
        assert self.agent_root is not None

    def save_tr_va_plots(self, tr_mean, val_mean, labels, name):

        # get the colors
        colors = ColorCollection.get_colors()

        # Create first plot
        plt.clf()
        fig_error = plt.figure("Error")
        fig_error.set_size_inches(15, 8)
        plt.title("Training Reward")
        plt.axhline(y=1, color='r', linestyle=':', label='Optimal')

        # plot using the correct colors
        for i in range(np.size(tr_mean, axis=1)):
            handles_top = plt.plot(tr_mean[:, i], color=colors[i][0], label=labels[i])

        plt.legend()
        plt.savefig(path.join(self.root, name))
