import matplotlib.pyplot as plt
import numpy as np
import os

from collection import ColorCollection

# create base folders
base_folder = '/home/markus/Git/BT/Experiments/MountainCar-v0/'
infixes = ['dropout', 'zoneout', 'shakeout']
#merge_folders = ['ddqn_reg_{}_15_0', 'ddqn_reg_{}_15_0','ddqn_reg_{}_15_0']
merge_folders = ['ddqn_reg_{}_ucb_15_8', 'ddqn_reg_{}_ucb_15_8','ddqn_reg_{}_ucb_15_8']
#merge_folders = ['ddqn_{}_15_1', 'ddqn_{}_15_2', 'ddqn_{}_15_1','ddqn_{}_15_2']
#merge_folders = ['ddqn_{}_15_1', 'ddqn_{}_15_2']
W = 50
show = False
running_avg = True
max_episode = 2500 - int(running_avg) * 2 * W
sigma = 15
exp_decay = 0.99
use_kernel = 'exp'
colors = [  #['#af11a5', '#af11a580'],  # lila
            ['#ff1000', '#ff230080'],  # red
            ['#007d1c', '#00b35880'],  # green
            ['#1142aa', '#057d9f80'],  # blue
            ['#ffe500', '#ffe50080'],  # yellow
            ['#ff8100', '#ff810080'],  # orange
            ['#4d10a3', '#4d10a380']]  # lila
pt_label_offset = 6
shift = 320
indices = None

out_folder = '/home/markus/Git/BT/Thesis/img/Evaluations/mountaincar'
file_name = 'mc_ucb_reg_ens_all_15.eps'

# create figure
fig = plt.figure(dpi=300)
plt.clf()
#ratio = 3.2
ratio = 2.5
x = 8.26 # full width
x = 0.8 * x
o = 6
pt_height = 0.76 * (x / ratio) * 72 # to points
fig.set_size_inches(x, x / ratio)

best_rewards = np.ones([len(infixes)]) * -201
if np.ndim(best_rewards) == 0: best_rewards = np.expand_dims(best_rewards, 0)
eff_indices = (indices if indices is not None else range(len(infixes)))
for i in eff_indices:
    infix = infixes[i]
    run = merge_folders[i].format(infix)

    comp_path = os.path.join(base_folder, run, "rewards.npy")
    rewards = np.load(comp_path)
    mean_rewards = np.mean(rewards, axis=0)
    var_rewards = np.var(rewards, axis=0)
    offset = 1.96 * np.sqrt(var_rewards / len(rewards))

    N = 2 * W + 1
    dx = max_episode/N
    gx = np.arange(-W, W + 1)
    steps = W - np.arange(0, N)

    if use_kernel == 'gaussian':
        kernel = np.exp(-(gx/sigma)**2/2)
    elif use_kernel == 'exp':
        kernel = exp_decay ** steps
    elif use_kernel == 'uniform':
        kernel = np.ones((N,))

    kernel[W+1:] = 0
    kernel = kernel / np.sum(kernel)
    #running_avg_mean_rewards = np.convolve(mean_rewards, np.ones((N,)) / N, mode='valid')
    #running_avg_var_rewards = np.convolve(var_rewards, np.ones((N,)) / N ** 2, mode='valid')

    running_avg_rewards = np.empty([len(rewards), max_episode])
    for k in range(len(rewards)):
        running_avg_rewards[k] = np.convolve(rewards[k], kernel, mode='valid')

    running_avg_mean_rewards = np.mean(running_avg_rewards, axis=0)
    running_avg_var_rewards = np.var(running_avg_rewards, axis=0)
    running_avg_offset = 1.96 * np.sqrt(running_avg_var_rewards / len(running_avg_var_rewards))

    print("{} has max {}".format(infixes[i], np.max(rewards)))
    print("{} has min {}".format(infixes[i], np.min(rewards)))
    print("{} has mean max {}".format(infixes[i], np.max(mean_rewards)))
    print("{} has mean min {}".format(infixes[i], np.min(mean_rewards)))
    print("{} has runavg max {}".format(infixes[i], np.max(running_avg_mean_rewards)))
    print("{} has runavg min {}".format(infixes[i], np.min(running_avg_mean_rewards)))

    best_episode = np.argmax(mean_rewards)
    print("{} hbest mean episode is {}".format(infixes[i], best_episode))

    best_episode = np.argmax(np.max(rewards, 0))
    print("{} best single episode is {}".format(infixes[i], best_episode))

    best_episode = np.argmax(running_avg_mean_rewards)
    print("{} best runavg episode is {}".format(infixes[i], best_episode))

    if running_avg:
        arng = np.arange(W, W + len(running_avg_mean_rewards))
        plt.fill_between(arng, running_avg_mean_rewards - running_avg_offset, running_avg_mean_rewards + running_avg_offset, facecolor=colors[i][1])
        plt.plot(arng, running_avg_mean_rewards, label="M", color=colors[i][0])

        best_rewards[i] = (np.max(running_avg_mean_rewards))

        print("mean reward is {}".format(np.mean(running_avg_mean_rewards)))
    else:
        #plt.fill_between(np.arange(len(mean_rewards)), mean_rewards - offset, mean_rewards + offset, facecolor=colors[i][1])
        plt.plot(mean_rewards, label="M", color=colors[i][0])
        print("mean reward is {}".format(np.mean(mean_rewards)))

    plt.xlabel("t")
    plt.tight_layout()

y_plot_int = (np.max(best_rewards) + 200) / (1 - 2 / pt_height * o)
pt_offset_ratio = y_plot_int / pt_height
o = pt_offset_ratio * o
upper_y = np.max(best_rewards) + pt_offset_ratio * (pt_label_offset + 10 + 6 + 3)
s_best_rewards = np.flip(np.argsort(-best_rewards, axis=0)[:len(eff_indices)], axis=0)
if np.ndim(s_best_rewards) == 0: s_best_rewards = np.expand_dims(s_best_rewards, 0)
for ei in range(len(eff_indices)):
    ri = eff_indices[s_best_rewards[ei]]
    plt.axhline(y=best_rewards[ri], color=colors[ri][0], linestyle=':')
    plt.text(ei * shift, best_rewards[ri] + pt_offset_ratio * (pt_label_offset + 10 + 6), str(round(best_rewards[ri], 2)), color=colors[ri][0],
             horizontalalignment='left', verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3})

plt.ylim([-200 - o, upper_y + o])
save_path = os.path.join(out_folder, file_name.format(infix))
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print("Saved to {}".format(save_path))
if show: plt.show()