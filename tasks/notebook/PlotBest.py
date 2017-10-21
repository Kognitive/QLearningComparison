from collection.ColorCollection import ColorCollection
from collection.PolicyCollection import PolicyCollection

folder = '/home/markus/Git/BT/Evaluations/runs/grid_world_5/'

import numpy as np
import matplotlib.pyplot as plt

from os import path
from matplotlib import rc
rc('font',**{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

# define the names which you want to be plotted
root_folder = "/home/markus/Git/BT/Experiments/MDPs"
out_folder = "/home/markus/Git/BT/Thesis/img/Evaluations/"
file_name = "old_strategies"
#root_folder = "/home/markus/Git/BT/Experiments/presi/"
#root_folder = "/home/markus/Git/BT/Experiments/unmerged/"
env_folders = [["grid_world_10", 1500, 0], ["exp_chain_50", 150, 0], ["shared_chain_33", 600, 0], ["bin_flip_6", 600, 0.5], ["deep_sea_10", 1000, 0.5]]
#plot_best_names = [["cb_pseudo_count", 0], ["bootstrapped", 1], ["ucb_infogain", 0], ["shared_bootstrap", 0]]
plot_best_names = [["eps_greedy", 0], ["boltzmann", 0], ["optimistic", 0], ["ucb", 0]]
use_best_at_train = True
cut_at_min = -1
show_best = 1
plot_both = False
plot_variance = True
save_fig = True
# matrix for best rewards and list for best labels
correct = False
use_cumulative = True
best_tr_rewards = None
best_va_rewards = None
best_tr_var_rewards = None
best_labels = list()
n = 0
m = len(plot_best_names)
for [env_folder, cut_at, min_lim] in env_folders:
    print("-" * 40)
    print(env_folder)
    best_tr_rewards = None
    # get environemnt path and iterate over all agents which should be plotted
    env_path = path.join(root_folder, env_folder)
    for j in range(m):

        # define the reward tensors
        i = j * show_best
        name = plot_best_names[j][0]
        start = plot_best_names[j][1]
        batch = PolicyCollection.get_batch(name)
        agent_path = path.join(env_path, name)
        if plot_variance: tr_var_tensor = np.loadtxt(path.join(agent_path, "tr_rewards_var.np"))
        tr_tensor = np.loadtxt(path.join(agent_path, "tr_rewards_mean.np"))[:1500]
        tr_tensor[0] = tr_tensor[1]
        tr_tensor = tr_tensor if np.ndim(tr_tensor) == 2 else np.expand_dims(tr_tensor, 1)
        if plot_variance:
            tr_var_tensor = tr_var_tensor if np.ndim(tr_var_tensor) == 2 else np.expand_dims(tr_var_tensor, 1)

        # init best rewards
        if best_tr_rewards is None:
            n = np.minimum(np.size(tr_tensor, 0), np.maximum(cut_at, cut_at_min))
            best_tr_rewards = np.empty((n, m * show_best))
            if plot_variance: best_tr_var_rewards = np.empty((n, m * show_best))
            best_va_rewards = np.empty((n, m * show_best))

        # check whether to use the cumulative
        arr = np.sum(tr_tensor, axis=0) if use_cumulative else tr_tensor[-1]

        # get the best indices
        best_indices = np.argmax(arr[start:]) + start
        best_indices = best_indices if np.ndim(best_indices) == 1 else np.expand_dims(best_indices, 0)
        best_indices = best_indices[-show_best:]

        # get best rewards
        best_tr_rewards[:, i:i+show_best] = tr_tensor[:n, best_indices]
        if plot_variance: best_tr_var_rewards[:, i:i+show_best] = tr_var_tensor[:n, best_indices]
        [best_labels.append(batch[l][0]) for l in best_indices]

    # get the colors as well
    colors = ColorCollection.get_colors()[1:]

    ratio = 11.5 / 4.2
    x = 2.5
    fig_error = plt.figure(0, dpi=300)
    plt.clf()
    fig_error.set_size_inches(x * ratio, x)

    if correct:
        best_tr_rewards = 2 * (best_tr_rewards - 0.5)
        best_tr_rewards = np.maximum(0.0, best_tr_rewards)

    plt.axhline(y=1, color='r', linestyle=':', label='Optimal')
    if min_lim == 0:
        plt.axhline(y=0, color='b', linestyle=':', label='Minimal')
    plt.ylim([-0.1 + min_lim, 1.1])
    for k in range(show_best * m):
        print(best_labels[k])
        plt.plot(best_tr_rewards[:n, k], color=colors[k][0], label=best_labels[k])

    if plot_variance:
        offset = 1.96 * np.sqrt(best_tr_var_rewards / 2500)
        for k in range(show_best * m):
            rng = np.arange(len(offset[:, k]))
            plt.fill_between(rng, best_tr_rewards[:, k] - offset[:, k], best_tr_rewards[:, k] + offset[:, k], facecolor=colors[k][1])

    plt.legend(bbox_to_anchor=(-0.09, 1), fontsize=15)
    plt.tight_layout()

    save_path = path.join(out_folder, env_folder, file_name + str(".pdf"))
    if save_fig: plt.savefig(save_path)
    else: plt.show()