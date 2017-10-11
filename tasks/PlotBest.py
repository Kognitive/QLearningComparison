from collection.ColorCollection import ColorCollection
from collection.PolicyCollection import PolicyCollection

folder = 'run/broadsearch_q_learning/2017-09-15_06-25-02/bin_flip_8/'

import numpy as np
import matplotlib.pyplot as plt

from os import path
from matplotlib import rc
rc('font',**{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

# define the names which you want to be plotted
root_folder = "run/TaskComparisonThesis"
env_folder = "grid_world_10"
plot_best_names = ["eps_greedy", "boltzmann", 'optimistic', 'ucb', "bootstrapped", "cb_pseudo_count"]
use_best_at_train = False
cut_at = 10000
show_best = 1
plot_both = False

# matrix for best rewards and list for best labels
use_cumulative = False
best_tr_rewards = None
best_va_rewards = None
best_labels = list()
n = 0
m = len(plot_best_names)

# get environemnt path and iterate over all agents which should be plotted
env_path = path.join(root_folder, env_folder)
for j in range(m):

    # define the reward tensors
    i = j * show_best
    name = plot_best_names[j]
    batch = PolicyCollection.get_batch(name)
    agent_path = path.join(env_path, name)
    va_tensor = np.loadtxt(path.join(agent_path, "va_rewards_mean.np"))
    tr_tensor = np.loadtxt(path.join(agent_path, "tr_rewards_mean.np"))
    tr_tensor = tr_tensor if np.rank(tr_tensor) == 2 else np.expand_dims(tr_tensor, 1)
    va_tensor = va_tensor if np.rank(va_tensor) == 2 else np.expand_dims(va_tensor, 1)

    # init best rewards
    if best_tr_rewards is None:
        n = np.minimum(np.size(tr_tensor, 0), cut_at)
        best_tr_rewards = np.empty((n, m * show_best))
        best_va_rewards = np.empty((n, m * show_best))

    # check whether to use the cumulative
    selected = tr_tensor if use_best_at_train else va_tensor
    arr = np.sum(selected, axis=0) if use_cumulative else selected[-1]

    # get the best indices
    best_indices = np.argpartition(arr, -show_best)
    best_indices = best_indices if np.rank(best_indices) == 1 else np.expand_dims(best_indices, 0)
    best_indices = best_indices[-show_best:]

    # get best rewards
    best_tr_rewards[:, i:i+show_best] = tr_tensor[:n, best_indices]
    best_va_rewards[:, i:i+show_best] = va_tensor[:n, best_indices]
    [best_labels.append(batch[l][0]) for l in best_indices]

# get the colors as well
colors = ColorCollection.get_colors()

fig_error = plt.figure(0)
fig_error.set_size_inches(15.0, 8.0)

if plot_both:

    top = fig_error.add_subplot(211)
    bottom = fig_error.add_subplot(212)
    top.axhline(y=1, color='r', linestyle=':', label='Optimal')
    bottom.axhline(y=1, color='r', linestyle=':', label='Optimal')
    top.set_title("On-Policy (Training)")
    bottom.set_title("Off-Policy (Validation)")
    #top.set_yscale("log", nonposy='clip')
    #bottom.set_yscale("log", nonposy='clip')

    for k in range(show_best * m):
        top.plot(best_tr_rewards[:, k], color=colors[k][0], label=best_labels[k])
        bottom.plot(best_va_rewards[:, k], color=colors[k][0], label=best_labels[k])

    bottom.legend()
    plt.show()
else:
    plt.axhline(y=1, color='r', linestyle=':', label='Optimal')
    # plt.xlim([0, n])
    # plt.suptitle("On-Policy (Training)")
    #top.set_yscale("log", nonposy='clip')
    #bottom.set_yscale("log", nonposy='clip')

    for k in range(show_best * m):
        plt.plot(best_tr_rewards[:, k], color=colors[k][0], label=best_labels[k])

    plt.legend(bbox_to_anchor=(-0.03, 1), fontsize=15)
    plt.tight_layout()
    plt.show()