import numpy as np
import matplotlib.pyplot as plt
from os import makedirs, path
from collection.ColorCollection import ColorCollection

class DirectoryManager:

    def __init__(self, directory):

        # check if path exists
        if not path.exists(directory):
            makedirs(directory)

        self.root = directory
        self.env_root = None
        self.agent_root = None

    def set_env_dir(self, name):
        assert self.root is not None

        # create env path
        env = path.join(self.root, name)

        # check if path exists
        if not path.exists(env):
            makedirs(env)

        self.env_root = env

    def set_agent_dir(self, name):
        assert self.env_root is not None

        # create env path
        ag = path.join(self.env_root, name)

        # check if path exists
        if not path.exists(ag):
            makedirs(ag)

        # create the output dirs
        for typ in []:#['densities', 'q_functions']:
            reward_dir = path.join(ag, typ)
            if not path.exists(reward_dir):
                makedirs(reward_dir)

        self.agent_root = ag

    def write_policy_params(self, policies):
        assert self.agent_root is not None

    def save_tr_va_plots(self, tr_mean, val_mean, labels, name):

        # get the colors
        colors = ColorCollection.get_colors()

        # Create first plot
        plt.clf()
        fig_error = plt.figure("Error")
        fig_error.set_size_inches(15, 8)
        top = fig_error.add_subplot(211)
        bottom = fig_error.add_subplot(212)
        top.set_title("Training Reward")
        bottom.set_title("Validation Reward")
        top.axhline(y=1, color='r', linestyle=':', label='Optimal')
        bottom.axhline(y=1, color='r', linestyle=':', label='Optimal')

        # plot using the correct colors
        for i in range(np.size(tr_mean, axis=1)):
            handles_top = top.plot(tr_mean[:, i], color=colors[i][0], label=labels[i])
            handles_bottom = bottom.plot(val_mean[:, i], color=colors[i][0], label=labels[i])

        top.legend()
        bottom.legend()
        plt.savefig(path.join(self.agent_root, name))
