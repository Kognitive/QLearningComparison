import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime
import os

from agents.QLearningAgent import QLearningAgent
from policies.GreedyPolicy import GreedyPolicy
from policies.BoltzmannPolicy import BoltzmannPolicy
from policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy
from environments.ExplorationChain import ExplorationChain
from environments.BinaryFlipEnvironment import BinaryFlipEnvironment
from environments.DeepSeaExploration import DeepSeaExploration
from environments.GridWorld import GridWorld

# the number of models an average should be taken
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ------------------------------ SETTINGS ------------------------------------
env_build = [[BinaryFlipEnvironment, "bin_flip", [4, 8, 12], lambda n: 2 * n],
             [DeepSeaExploration, "deep_sea", [5, 10, 20, 30, 40, 50], lambda n: n],
             [ExplorationChain, "exp_chain", [5, 10, 20, 30, 40, 50], lambda n: n + 9]]

color_pool = [
    ['#1455bc', '#81b0f945'],
    ['#e58d09', '#edc89045'],
    ['#ad0bbc', '#bb5ec445'],
    ['#56b21c', '#b4db9b45'],
    ['#870404', '#d1555545'],
    ['#c4c417', '#cece4045'],
    ['#7fbc16', '#b6e56445'],
    ['#0c9994', '#6ee5e145'],
    ['#0f117a', '#5e60e545'],
    ['#e520db', '#ce61c845'],
]

# define the policy batch size
policy_batch_size = 20

# define the different policies you want to try out
policy_batches = [
    #["eps_greedy", EpsilonGreedyPolicy,
    #   {'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]}],
    #["boltzmann", BoltzmannPolicy,
    #    {'temperature': [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 1, 10, 50, 100]}],
    ["deterministic_cb_pseudo_count", GreedyPolicy,
     {'pseudo_count': [True], 'pseudo_count_type': ['count_based'], 'optimistic': [True, False],
      'beta': [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 5, 10]}],

    ["deterministic_bootstrapped", GreedyPolicy,
     {'num_heads': [1, 3, 5, 7, 10, 15, 20]}],

    ["deterministic_bootstrapped_cb_pseudo_count", GreedyPolicy,
        {'pseudo_count': [True], 'num_heads': [3, 5, 7], 'pseudo_count_type': ['count_based'], 'optimistic': [True, False], 'beta': [0.1, 0.5, 1, 5]}],

     #["eps_greedy_bootstrapped", EpsilonGreedyPolicy,
    #    {'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5], 'num_heads': [1, 3, 5, 7, 10, 15, 20]}],
    ["boltzmann_bootstrapped", BoltzmannPolicy,
        {'temperature': [0.005, 0.05, 0.3, 1], 'num_heads': [3, 5, 7]}],

   #["deterministic_pg_pseudo_count", GreedyPolicy,
    # {'pseudo_count': [True], 'pseudo_count_type': ['prediction_gain'], 'optimistic': [True, False],
    #  'beta': [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 5, 10], 'c': [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 5, 10]}],
    #["deterministic_pc_pseudo_count", GreedyPolicy,
    #     {'pseudo_count': [True], 'pseudo_count_type': ['pseudo_count'], 'optimistic': [True, False],
    #      'beta': [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 5, 10]}],


    ["deterministic_ucb", GreedyPolicy,
        {'optimistic': [True, False], 'ucb': [True], 'p': [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 5, 10]}]
    ]

batch_num = 0

# define a list for the best policies
best_va_policies = len(policy_batches) * [0]
best_tr_policies = len(policy_batches) * [0]

# create variable for the steps and do this amount of steps.
num_models = 500
show_models = 3
num_episodes = 1500
learned_episodes = 200

save_density = True
density_episodes = 50

# control if the status should be displayed
status = True
print_policy = False
plot_conf_interval = False
live_plots = True
regret = False
plot_max_policy = False
plot_evaluation = True
plot_q_function = True
plot_densities = True
normalize_by_optimal = True

# create timestamp
out_dir = 'run/broadsearch_q_learning/'
timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
out_dir = out_dir + timestamp + '/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# iterate over all environments
for [env_build, env_name, range_N, length] in env_build:
    for n in range_N:
        num_steps = length(n)

        # use environment dir for graphs
        env_dir = out_dir + env_name + '_' + str(n) + '/'
        if not os.path.exists(env_dir):
            os.makedirs(env_dir)
            os.makedirs(env_dir + 'npy_rewards')
            os.makedirs(env_dir + 'plots_rewards')
            os.makedirs(env_dir + 'plots_density')

        batch_num = 0
        sub_batch_num = 0
        params = policy_batches[batch_num][2]
        policy = policy_batches[batch_num][1]
        name = policy_batches[batch_num][0]
        best_cum_va_rew = - 2 ** 32
        best_cum_tr_rew = - 2 ** 32

        keys = [key for key in params]
        batch_counter = [0 for key in keys]
        batch_length = [len(params[key]) for key in keys]
        len_keys = len(keys) - 1

        # iterate over all batches
        while batch_num < (len(policy_batches) + 2):

            batch = list()
            batch_complete = False
            if batch_num < len(policy_batches):
                while not batch_complete and len(batch) < policy_batch_size:

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

            elif batch_num == len(policy_batches):
                batch = best_va_policies
                name = "best_va"
            elif batch_num == len(policy_batches) + 1:
                batch = best_tr_policies
                name = "best_tr"

            tf.reset_default_graph()
            graph = tf.Graph()
            with graph.as_default():
                with tf.Session(graph=graph) as sess:

                    env = env_build("test", [num_models], n)
                    state_space = env.state_space
                    action_space = env.action_space
                    log_action_size = action_space.get_log2_size()

                    # --------------------- Determine the optimal reward --------------------

                    # Determine the agent count
                    num_policies = len(batch)
                    optimal_ih_rew, min_q, max_q = env.get_optimal(num_steps, 0.99)

                    # --------------------------------------------------------------------------

                    # Iterate over all policies and create an agent using that specific policy
                    agents = list()
                    environments = list()
                    for pol_num in range(num_policies):

                        # Get policy and unqiue name
                        pe = batch[pol_num]
                        unique_name = str(pol_num)

                        # extract important fields
                        policy = pe[2]
                        policy_config = pe[3]
                        policy_config['num_models'] = num_models
                        policy_config['action_space'] = action_space
                        policy_config['min_q'] = min_q
                        policy_config['max_q'] = max_q
                        policy_config['discount'] = 0.99

                        current_env = env_build(unique_name, [num_models], n)
                        environments.append(current_env)
                        agents.append(QLearningAgent(sess, unique_name, current_env, policy, policy_config))

                    # init variables
                    init = tf.global_variables_initializer()
                    sess.run(init)

                    # define the evaluation rewards
                    training_rewards = np.empty((num_episodes + 1, num_policies, num_models))
                    training_mean = np.empty((num_episodes + 1, num_policies))
                    training_var = np.empty((num_episodes + 1, num_policies))

                    # set value for first episode
                    training_rewards[0, :, :] = 0
                    training_mean[0, :] = 0
                    training_var[0, :] = 0

                    # define the evaluation rewards
                    val_rewards = np.empty((num_episodes + 1, num_policies, num_models))
                    val_mean = np.empty((num_episodes + 1, num_policies))
                    val_var = np.empty((num_episodes + 1, num_policies))

                    # set value for first episode
                    val_rewards[0, :, :] = 0
                    val_mean[0, :] = 0
                    val_var[0, :] = 0

                    # get the learn operations

                    # retrieve the learn operations
                    update_and_receive_rewards = [agent.q_tensor_update for agent in agents]
                    perform_ops = [agent.apply_actions for agent in agents]

                    reset_ops = [envs.reset_op for envs in environments]
                    cum_rew_ops = [envs.cum_rewards for envs in environments]
                    episode = 0

                    # iterate over episodes
                    for episode in range(1, num_episodes + 1):
                        print("Environment={}, N={}, BatchNum={}, SubBatchNum={}, Episode={}".format(env_name, n, batch_num, sub_batch_num, episode))
                        # reset all environments
                        sess.run(reset_ops)

                        # for each agent sample a new head
                        state_dict = {}
                        for k in range(num_policies):
                            agents[k].sample_head()
                            state_dict[agents[k].use_best] = False

                        # repeat this for the number of steps
                        for k in range(num_steps):

                            # receive rewards and add
                            sess.run(update_and_receive_rewards, feed_dict=state_dict)

                        # copy values
                        training_rewards[episode, :, :] = sess.run(cum_rew_ops) / optimal_ih_rew

                        # determine mean and variance
                        training_mean[episode, :] = np.mean(training_rewards[episode, :, :], axis=1)
                        training_var[episode, :] = np.var(training_rewards[episode, :, :], axis=1)

                        # reset all environments
                        sess.run(reset_ops)

                        # for each agent sample a new head
                        state_dict = {}
                        for k in range(num_policies):
                            agents[k].sample_head()
                            state_dict[agents[k].use_best] = True

                        # repeat this for the number of steps
                        for k in range(num_steps):

                            # Execute all actions and collect rewards
                            sess.run(perform_ops, feed_dict=state_dict)

                        # copy values
                        val_rewards[episode, :, :] = sess.run(cum_rew_ops) / optimal_ih_rew

                        # determine mean and variance
                        val_mean[episode, :] = np.mean(val_rewards[episode, :, :], axis=1)
                        val_var[episode, :] = np.var(val_rewards[episode, :, :], axis=1)

                        if episode > learned_episodes:
                            important_episodes = val_mean[(episode - learned_episodes + 1):episode + 1, :]
                            mean_important_episodes = np.mean(important_episodes, axis=1)
                            if np.min(mean_important_episodes, axis=0) >= 0.99:
                                break

                        #if save_density and (episode - 1) % density_episodes == 0:

            # save the collected reward
            np.save(env_dir + 'npy_rewards/tr_' + name + "_" + str(sub_batch_num) + '_mean.npy', training_mean)
            np.save(env_dir + 'npy_rewards/tr_' + name + "_" + str(sub_batch_num) + '_var.npy', training_var)
            np.save(env_dir + 'npy_rewards/va_' + name + "_" + str(sub_batch_num) + '_mean.npy', val_mean)
            np.save(env_dir + 'npy_rewards/va_' + name + "_" + str(sub_batch_num) + '_var.npy', val_var)
            plt_path = env_dir

            # determine the best policy
            if batch_num < len(policy_batches):

                # build the cumulative reward on the validation set
                cum_reward_tr = np.sum(training_mean, axis=0)
                best_index_tr = np.argmax(cum_reward_tr)
                best_value_tr = np.max(cum_reward_tr)
                if best_value_tr > best_cum_tr_rew:
                    best_tr_policies[batch_num] = batch[best_index_tr]
                    best_cum_tr_rew = best_value_tr

                # build the cumulative reward on the validation set
                cum_reward_va = np.sum(val_mean, axis=0)
                best_index_va = np.argmax(cum_reward_va)
                best_value_va = np.max(cum_reward_va)
                if best_value_va > best_cum_va_rew:
                    best_va_policies[batch_num] = batch[best_index_va]
                    best_cum_va_rew = best_value_va

                plt_path = env_dir + 'plots_rewards/'

            # Create the figure containing the training and validation error
            plt.clf()
            fig_error = plt.figure(0)
            fig_error.set_size_inches(15.0, 8.0)
            top = fig_error.add_subplot(211)
            bottom = fig_error.add_subplot(212)

            top.set_title("Training Error")
            bottom.set_title("Validation Error")
            top.axhline(y=1, color='r', linestyle=':', label='Optimal')
            bottom.axhline(y=1, color='r', linestyle=':', label='Optimal')

            # plot using the correct colors
            for i in range(np.size(training_mean, axis=1)):

                tr_confidence_offset = 1.96 * training_var[:episode] / np.sqrt(np.size(training_mean, axis=1))
                va_confidence_offset = 1.96 * val_var[:episode] / np.sqrt(np.size(val_mean, axis=1))
                n_range = np.arange(0, episode)

                tr_lower = training_mean[:episode] - tr_confidence_offset
                tr_upper = training_mean[:episode] + tr_confidence_offset
                #top.fill_between(n_range, tr_upper, tr_lower, color=color_pool[i][1])
                #bottom.fill_between(n_range, val_mean[:episode] - va_confidence_offset, val_mean[:episode] + va_confidence_offset, color=color_pool[i][1])

                label = "{} ({})".format(batch[i][0], batch[i][1])
                handles_top = top.plot(training_mean[:episode, i], label=label)
                handles_bottom = bottom.plot(val_mean[:episode, i], label=label)

            #bottom.legend(loc='lower right')
            plt.savefig(plt_path + name + "_" + str(sub_batch_num) + '.eps')

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
            else:
                sub_batch_num += 1