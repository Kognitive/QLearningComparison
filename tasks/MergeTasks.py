import numpy as np
import os.path

merge_agents = ["eps_greedy", "boltzmann", "ucb", "optimistic"]

#merge_folders = [['/home/markus/Git/BT/Experiments/unmerged/', ['deep_sea_one_10', 'deep_sea_two_10', 'deep_sea_three_10', 'deep_sea_four_10']]]
merge_folders = [['/home/markus/Git/BT/Experiments/MDPs', ['deep_sea_one_19', 'deep_sea_two_19']]]

output_folder = '/home/markus/Git/BT/Experiments/MDPs/deep_sea_19/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

episodes = 5000
mult_unit = 750

for batch_names in merge_agents:

    tr_list = list()
    va_list = list()
    tr_var_list = list()
    va_var_list = list()
    for [folder, problems] in merge_folders:
        for pi in range(len(problems)):
            problem = problems[pi]
            agent_root = os.path.join(folder, problem, batch_names)

            off = 0 if pi == 0 or pi == 2 else 1
            # Retrieve the mean and add it to the list
            tr_mean = np.loadtxt(os.path.join(agent_root, "tr_rewards_mean.np")) + off
            tr_list.append(mult_unit * tr_mean)

            # Now merge also the variance
            tr_var_list.append((mult_unit * (np.loadtxt(os.path.join(agent_root, "tr_rewards_mean.np")) + tr_mean ** 2)))

    tr_sum = tr_list[0]
    tr_var_sum = tr_var_list[0]

    for i in range(1, len(tr_list)):
        tr_sum += tr_list[i]
        tr_var_sum += tr_var_list[i]

    tr_sum /= mult_unit * len(tr_list)
    tr_var_sum = tr_var_sum / (mult_unit * len(tr_list)) - tr_sum ** 2

    save_agent_root = os.path.join(output_folder, batch_names)
    if not os.path.exists(save_agent_root):
        os.makedirs(save_agent_root)
    np.savetxt(os.path.join(save_agent_root, "tr_rewards_mean.np"), tr_sum)
    np.savetxt(os.path.join(save_agent_root, "tr_rewards_var.np"), tr_var_sum)