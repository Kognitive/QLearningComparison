import numpy as np
import os.path

merge_agents = ['boltzmann', 'bootstrapped', 'cb_pseudo_count', 'eps_greedy', 'optimistic', 'ucb']

merge_folders = [['run/Notebook/GPU/', ['deep_sea_three_20', 'deep_sea_four_20']],
                 ['run/Computer/GPU/', ['deep_sea_one_20', 'deep_sea_two_20']]]

output_folder = 'run/Merged/deep_sea_20'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

episodes = 5000
mult_unit = 625

for batch_names in merge_agents:

    tr_list = list()
    va_list = list()
    for [folder, problems] in merge_folders:
        for problem in problems:
            agent_root = os.path.join(folder, problem, batch_names)

            tr_list.append(mult_unit * np.loadtxt(os.path.join(agent_root, "tr_rewards_mean.np")))
            va_list.append(mult_unit * np.loadtxt(os.path.join(agent_root, "va_rewards_mean.np")))

    tr_sum = tr_list[0]
    va_sum = va_list[0]

    for i in range(1, len(tr_list)):
        tr_sum += tr_list[i]
        va_sum += va_list[i]

    tr_sum /= mult_unit * len(tr_list)
    va_sum /= mult_unit * len(va_list)

    save_agent_root = os.path.join(output_folder, batch_names)
    if not os.path.exists(save_agent_root):
        os.makedirs(save_agent_root)
    np.savetxt(os.path.join(save_agent_root, "tr_rewards_mean.np"), tr_sum)
    np.savetxt(os.path.join(save_agent_root, "va_rewards_mean.np"), va_sum)