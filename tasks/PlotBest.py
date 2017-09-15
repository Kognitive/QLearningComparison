folder = 'run/broadsearch_q_learning/2017-09-15_06-25-02/bin_flip_8/'

import numpy as np
import matplotlib.pyplot as plt
from policies.GreedyPolicy import GreedyPolicy
from policies.BoltzmannPolicy import BoltzmannPolicy
from policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy

# define the different policies you want to try out
policy_batches = [
    ["eps_greedy", EpsilonGreedyPolicy,
       {'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]}],
    ["boltzmann", BoltzmannPolicy,
        {'temperature': [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 1, 10, 50, 100]}],
    ["deterministic_cb_pseudo_count", GreedyPolicy,
     {'pseudo_count': [True], 'pseudo_count_type': ['count_based'], 'optimistic': [False],
      'beta': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10]}],
    ["deterministic_cb_pseudo_count_optimistic", GreedyPolicy,
     {'pseudo_count': [True], 'pseudo_count_type': ['count_based'], 'optimistic': [False],
      'beta': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10]}],

    ["deterministic_bootstrapped", GreedyPolicy,
     {'num_heads': [1, 3, 5, 7, 10, 15]}],
    ["deterministic_bootstrapped_cb_pseudo_count", GreedyPolicy,
        {'pseudo_count': [True], 'num_heads': [3, 5, 7], 'pseudo_count_type': ['count_based'], 'optimistic': [False], 'beta': [0.005, 0.01, 0.05]}],
    ["deterministic_bootstrapped_cb_pseudo_count_optimistic", GreedyPolicy,
        {'pseudo_count': [True], 'num_heads': [3, 5, 7], 'pseudo_count_type': ['count_based'], 'optimistic': [True], 'beta': [0.005, 0.01, 0.05]}],

     ["eps_greedy_bootstrapped", EpsilonGreedyPolicy,
        {'epsilon': [0.01, 0.05, 0.1], 'num_heads': [3, 5, 7]}],

    ["boltzmann_bootstrapped", BoltzmannPolicy,
        {'temperature': [0.005, 0.05, 0.3], 'num_heads': [3, 5, 7]}],

    ["deterministic_pc_pseudo_count, ", GreedyPolicy,
        {'pseudo_count': [True], 'pseudo_count_type': ['pseudo_count'], 'optimistic': [False], 'beta': [0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]}],

    ["deterministic_pc_pseudo_count_optimistic, ", GreedyPolicy,
        {'pseudo_count': [True], 'pseudo_count_type': ['pseudo_count'], 'optimistic': [True], 'beta': [0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]}],

    ["deterministic_ucb", GreedyPolicy,
        {'optimistic': [True], 'ucb': [True], 'p': [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 5, 10]}],

    ["deterministic_ucb", GreedyPolicy,
        {'optimistic': [True], 'ucb': [False], 'p': [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 5, 10]}]
    ]

fig_error = plt.figure(0)
fig_error.set_size_inches(15.0, 8.0)

top = fig_error.add_subplot(211)
bottom = fig_error.add_subplot(212)
top.axhline(y=1, color='r', linestyle=':', label='Optimal')
bottom.axhline(y=1, color='r', linestyle=':', label='Optimal')
batch_num = 0
params = policy_batches[batch_num][2]

keys = [key for key in params]
batch_counter = [0 for key in keys]
batch_length = [len(params[key]) for key in keys]
len_keys = len(keys) - 1

name = policy_batches[batch_num][0]

policy = policy_batches[batch_num][1]

while batch_num < (len(policy_batches)):
    batch = list()
    batch_complete = False
    while not batch_complete:

        pol_config = {}
        param_string = ""
        for key_num in range(len(keys)):
            key = keys[key_num]
            pol_config[key] = params[key][batch_counter[key_num]]
            param_string += "{}={}".format(key, pol_config[key])
            if key_num < len(keys) - 1:
                param_string += ", "

        # append to current batch
        batch.append([name, param_string, policy, pol_config])

        i = len_keys
        batch_counter[i] += 1
        while batch_counter[i] >= batch_length[i]:
            if i == 0:
                batch_complete = True
                break

            batch_counter[i] = 0
            batch_counter[i - 1] += 1
            i -= 1
    try:
        training_mean = np.load(folder + 'npy_rewards/tr_' + name + "_" + str(0) + '_mean.npy')
        training_var = np.load(folder + 'npy_rewards/tr_' + name + "_" + str(0) + '_var.npy')
        val_mean = np.load(folder + 'npy_rewards/va_' + name + "_" + str(0) + '_mean.npy')
        val_var = np.load(folder + 'npy_rewards/va_' + name + "_" + str(0) + '_var.npy')

        cum_val_mean = np.sum(val_mean, axis=0)
        hi = np.argmax(cum_val_mean)
        if val_mean[-1, hi] > 0.3:
            label = "{} ({})".format(batch[i][0], batch[i][1])
            top.plot(training_mean[:, hi], label=label)
            bottom.plot(training_mean[:, hi])
            top.legend()
    except:
        print()
    if batch_num >= len(policy_batches):
        batch_num += 1
        continue

    if batch_complete:
        batch_num += 1
        sub_batch_num = 0

        if batch_num >= len(policy_batches):
            continue

        params = policy_batches[batch_num][2]
        policy = policy_batches[batch_num][1]
        name = policy_batches[batch_num][0]

        best_cum_va_rew = - 2 ** 32
        best_cum_tr_rew = - 2 ** 32

        keys = [key for key in params]
        batch_counter = [0 for key in keys]
        batch_length = [len(params[key]) for key in keys]
        len_keys = len(keys) - 1

plt.show()