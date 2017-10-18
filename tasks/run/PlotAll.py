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
root_folder = "/home/markus/Git/BT/Thesis/img/Evaluations/runs/"
#root_folder = "/home/markus/Git/BT/Experiments/presi/"
#root_folder = "/home/markus/Git/BT/Experiments/unmerged/"
out_folder = "/home/markus/Git/BT/Thesis/img/Evaluations/"
problem_names = ["shared_chain_33", "exp_chain_50", "deep_sea_10", "grid_world_5", "bin_flip_4"]
cut_ats = [500, 100, 500, 1200, 200]
cut_min = -1
plot_best_names = "shared_bootstrap"
use_best_at_train = False
show_best = 1
plot_both = False
plot_variance = True
# matrix for best rewards and list for best labels
use_cumulative = False
best_tr_rewards = None
best_va_rewards = None
best_tr_var_rewards = None
best_labels = list()
n = 0
start = 1
m = len(plot_best_names)
for pindex in range(len(problem_names)):
    problem_name = problem_names[pindex]

    cut_at = np.maximum(cut_ats[pindex], cut_min)

    # get environemnt path and iterate over all agents which should be plotted
    env_path = path.join(root_folder, problem_name)
    name = plot_best_names
    batch = PolicyCollection.get_batch(name)
    agent_path = path.join(env_path, name)
    tr_tensor = np.loadtxt(path.join(agent_path, "tr_rewards_mean.np"))
    tr_var_tensor = np.loadtxt(path.join(agent_path, "tr_rewards_var.np"))
    tr_tensor = tr_tensor if np.rank(tr_tensor) == 2 else np.expand_dims(tr_tensor, 1)
    tr_var_tensor = tr_var_tensor if np.rank(tr_var_tensor) == 2 else np.expand_dims(tr_var_tensor, 1)

    [best_labels.append(batch[l][0]) for l in range(start, len(batch))]

    # get the colors as well
    colors = ColorCollection.get_colors()

    ratio = 11.5 / 4.2
    x = 2.5
    fig_error = plt.figure(0)
    plt.clf()
    fig_error.set_size_inches(x * ratio, x)

    #best_tr_rewards = 2 * (best_tr_rewards - 0.5)
    #best_tr_rewards = np.maximum(0.0, best_tr_rewards)

    plt.axhline(y=1, color='r', linestyle=':', label='Optimal')
    # plt.xlim([0, n])
    # plt.suptitle("On-Policy (Training)")
    #top.set_yscale("log", nonposy='clip')
    #bottom.set_yscale("log", nonposy='clip')
    print(problem_name)
    bpi = np.argmax(np.sum(tr_tensor, axis=0)) - 1
    print(best_labels[bpi])
    for k in range(len(batch) - start):
        plt.plot(tr_tensor[:cut_at, start+k], color=colors[k][0], label=best_labels[k])

    if plot_variance:
        offset = 1.96 * np.sqrt(tr_var_tensor[:cut_at, start+k] / 2500)
        for k in range(len(batch) - start):
            rng = np.arange(len(offset))
            plt.fill_between(rng, tr_tensor[:cut_at, start+k] - offset, tr_tensor[:cut_at, start+k] + offset, facecolor=colors[k][1])

    plt.legend(bbox_to_anchor=(-0.09, 1), fontsize=15)
    plt.tight_layout()

    outpath = path.join(out_folder, problem_name, plot_best_names + str(".eps"))
    plt.savefig(outpath)