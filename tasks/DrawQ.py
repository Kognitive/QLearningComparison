import matplotlib.pyplot as plt
import numpy as np
import os

# create base folders
base_folder = '/home/markus/Git/BT/Experiments/MountainCar-v0/'
infixes = ['dropout', 'zoneout', 'shakeout']
#merge_folders = ['ddqn_{}_15_1', 'ddqn_{}_15_2']
merge_folders = ['ddqn_reg_{}_15_0']
plot_q_funcs = True
max_episode = 2500
save = False
colors = [["#215ab7", "#215ab786"], ["#b70000", "#b7000045"]]

out_folder = '/home/markus/Git/BT/Thesis/img/Evaluations/mountaincar'
file_name = 'mc_{}_15_1_2.eps'
q_func_name = 'value_func_mc_ens_reg_{}.eps'

for ri in range(len(merge_folders)):

    # get the style params
    vtopmargin = 0.08
    vbottommargin = 0.2
    vmargin = 0.2
    hleftmargin = 0.12
    hrightmargin = 0.08
    hmargin = 0.05
    bmarginoff = 0.05
    bar_width = 0.015
    box = [-1.2, 0.6, -0.07, 0.07]
    cbox = list(box)

    # define the ticks of the first subplot
    steps = 0.6
    box_width = box[1] - box[0]
    y_ticks = box_width / steps
    first_ticks = np.round(np.arange(box[0], box[1], steps), 1)

    # get all ticks
    ticks = np.copy(first_ticks)
    labels = [str(el) for el in first_ticks]

    # know combine them to all ticks
    for i in range(1, len(infixes)):
        ticks = np.concatenate((ticks, np.add(first_ticks, i * box_width)), axis=0)

        labels.append("{}".format(first_ticks[0]))
        for el in first_ticks[1:]:
            labels.append(str(el))

    labels.append(np.round(first_ticks[-1] + steps, 1))
    ticks = np.concatenate((ticks, [box[0] + len(infixes) * box_width]))

    fig = plt.figure("q_funcs_{}".format(ri))
    plt.clf()
    complete_width = 8.3
    plot_width = (1 - bar_width - hleftmargin - hrightmargin - hmargin) * complete_width
    ratio = 1.0
    height = ratio * (plot_width / len(infixes)) / (1 - vtopmargin - vbottommargin)
    fig.set_size_inches(complete_width, height)

    q_funcs = list()

    for i in range(len(infixes)):
        run = merge_folders[ri].format(infixes[i])
        comp_path = os.path.join(base_folder, run, "q_funcs/q_2400.npy")
        q_func = np.load(comp_path)
        q_funcs.append(q_func)

    value_function = [np.max(q_func, axis=0) for q_func in q_funcs]
    merged_image = np.concatenate(value_function, axis=1)
    cbox[1] = box[1] + (len(infixes) - 1) * (box[1] - box[0])

    ax = fig.add_axes([hleftmargin, vbottommargin, (1 - bar_width - hleftmargin - hrightmargin - hmargin), 1 - vtopmargin - vbottommargin])
    vf = plt.imshow(merged_image, interpolation='nearest', extent=cbox, aspect='auto')
    ax.set_ylabel("v")
    ax.set_xlabel("x")
    plt.xticks(ticks, labels)

    for i in range(1, len(infixes)):
        plt.axvline(x=box[0] + i * (box[1] - box[0]), color='black', linewidth=0.5)

    clb_ax = fig.add_axes([1 - bar_width - hrightmargin, bmarginoff + vbottommargin, bar_width, 1 - (vbottommargin + vtopmargin + 2 * bmarginoff)])
    fig.colorbar(vf, cax=clb_ax)

    save_path = os.path.join(out_folder, q_func_name.format(merge_folders[ri].format("")))
    if save: plt.savefig(save_path)
    else: plt.show()