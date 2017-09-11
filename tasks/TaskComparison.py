import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

from agents.QLearningAgent import QLearningAgent
from policies.GreedyPolicy import GreedyPolicy
from policies.BoltzmannPolicy import BoltzmannPolicy
from policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy
from environments.ExplorationChain import ExplorationChain
from environments.GridWorld import GridWorld
from environments.BinaryFlipEnvironment import BinaryFlipEnvironment

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# the number of models an average should be taken

# ------------------------------ SETTINGS ------------------------------------
N = 3
env_build = GridWorld

# create variable for the steps and do this amount of steps.
num_models = 100
show_models = 3
num_episodes = 2000
num_steps = 3 * N

graph = tf.Graph()
with graph.as_default():
    with tf.Session(graph=graph) as sess:

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
        normalize_by_optimal = False

        env = env_build("test", [num_models], N)
        state_space = env.state_space
        action_space = env.action_space
        log_action_size = action_space.get_log2_size()

        time_frame = 20
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

        # define the different policies you want to try out
        policies = [
            ['Bootstrapped Greedy', GreedyPolicy, {'action_space': action_space, 'num_heads': 20, 'heads_per_sample': 4}]
            #["eps_greedy", EpsilonGreedyPolicy, {'action_space': action_space, 'epsilon': 0.05, 'pseudo_count': True, 'pseudo_count_type': 'prediction_gain', 'beta': 1, 'c': 1}]
             #['Bootstrapped Greedy', GreedyPolicy, {'action_space': action_space, 'num_heads': 20, 'pseudo_count': True, 'pseudo_count_type': 'pseudo_count', 'beta': 1}],
             #['Greedy Policy', GreedyPolicy, {'action_space': action_space, 'num_heads': 1}],
             #['Bootstrapped Greedy with CB Model (1)', BoltzmannPolicy, {'temperature': 0.5, 'action_space': action_space, 'num_heads': 20, 'pseudo_count': True, 'pseudo_count_type': 'count_based', 'beta': 1}],
             #['Bootstrapped Greedy with CB Model (0.5)', BoltzmannPolicy, {'temperature': 0.5, 'action_space': action_space, 'num_heads': 20, 'pseudo_count': True, 'pseudo_count_type': 'count_based', 'beta': 0.5}],
             #['Bootstrapped Greedy with CB Model (3)', BoltzmannPolicy, {'temperature': 0.5, 'action_space': action_space, 'num_heads': 20, 'pseudo_count': True, 'pseudo_count_type': 'count_based', 'beta': 3}],
             #['Bootstrapped Greedy with CB Model (5)', BoltzmannPolicy, {'temperature': 0.5, 'action_space': action_space, 'num_heads': 20, 'pseudo_count': True, 'pseudo_count_type': 'count_based', 'beta': 5}],
             #['Bootstrapped Greedy with CB Model (10)', BoltzmannPolicy, {'temperature': 0.5, 'action_space': action_space, 'num_heads': 20, 'pseudo_count': True, 'pseudo_count_type': 'count_based', 'beta': 10}]

             #['$\epsilon$-Greedy (0.001)', GreedyPolicy, {'action_space': action_space, 'optimistic': True, 'pseudo_count': True, 'pseudo_count_type': 'prediction_gain', 'beta': 10, 'c': 10}],
             #['$\epsilon$-Greedy (0.001)', GreedyPolicy, {'action_space': action_space, 'optimistic': True, 'pseudo_count': True, 'pseudo_count_type': 'pseudo_count', 'beta': 10}]
             #['$\epsilon$-Greedy (0.01)', EpsilonGreedyPolicy, {'action_space': action_space, 'epsilon': 0.01, 'pseudo_count': True, 'beta': 10}],
             #['$\epsilon$-Greedy (0.1)', EpsilonGreedyPolicy, {'action_space': action_space, 'epsilon': 0.1, 'pseudo_count': True, 'beta': 10}]

            # ['Boltzmann (0.1)', BoltzmannPolicy, {'temperature': 0.1}],
            # ['Boltzmann (1)', BoltzmannPolicy, {'temperature': 1}],
            # ['Boltzmann (10)', BoltzmannPolicy, {'temperature': 10}],
            # ['Deterministic Bootstrapped (3)', GreedyPolicy, {'action_space': action_space, 'num_heads': 3}],
            # ['Deterministic Bootstrapped (7)', GreedyPolicy, {'action_space': action_space, 'num_heads': 7}],
            # ['Deterministic Bootstrapped (12)', GreedyPolicy, {'action_space': action_space, 'num_heads': 12}],
            # ['Deterministic Bootstrapped (2)', GreedyPolicy, {'action_space': action_space, 'num_heads': 12}],
            # ['$\epsilon$-Greedy Bootstrapped', EpsilonGreedyPolicy, {'action_space': action_space, 'epsilon': 0.1, 'num_heads': 3}],
            # ['$\epsilon$-Greedy Bootstrapped (0.05, 3)', EpsilonGreedyPolicy, {'action_space': action_space, 'epsilon': 0.05, 'num_heads': 7}],
            # ['$\epsilon$-Greedy Bootstrapped (0.1, 3)', EpsilonGreedyPolicy, {'action_space': action_space, 'epsilon': 0.1, 'num_heads': 3}],
            # ['$\epsilon$-Greedy Bootstrapped (0.05, 5)', EpsilonGreedyPolicy, {'action_space': action_space, 'epsilon': 0.05, 'num_heads': 20}],
            # ['$\epsilon$-Greedy Bootstrapped (0.1, 5)', EpsilonGreedyPolicy, {'action_space': action_space, 'epsilon': 0.1, 'num_heads': 20}],
            # ['Boltzmann Bootstrapped', BoltzmannPolicy, {'temperature': 3, 'num_heads': 5}],
            # ['Optimistic-Greedy', GreedyPolicy, {'optimistic': True}],
            # ['UCB1-Greedy', EpsilonGreedyPolicy, {'ucb': True, 'p': 2.0, 'action_space': action_space, 'epsilon': 0.3}],
            # ['Pseudo-Count-Greedy 1', GreedyPolicy, {'pseudo_count': True, 'beta': 1000}]
        ]

        # --------------------- Determine the optimal reward --------------------

        # Determine the agent count
        num_policies = len(policies)
        optimal_ih_rew = env.get_optimal(num_steps, 0.99)

        # --------------------------------------------------------------------------

        # Iterate over all policies and create an agent using that specific policy
        agents = list()
        environments = list()
        for pol_num in range(num_policies):

            # Get polic and unqiue name
            pe = policies[pol_num]
            unique_name = str(pol_num)

            # extract important fields
            policy = pe[1]
            policy_config = pe[2]
            policy_config['num_models'] = num_models

            current_env = env_build(unique_name, [num_models], N)
            environments.append(current_env)
            agents.append(QLearningAgent(sess, unique_name, current_env, policy, policy_config))

        # init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # define the evaluation rewards
        training_rewards = np.empty((num_episodes + 1, len(policies), num_models))
        training_mean = np.empty((num_episodes + 1, len(policies)))
        training_var = np.empty((num_episodes + 1, len(policies)))

        # set value for first episode
        training_rewards[0, :, :] = 0
        training_mean[0, :] = 0
        training_var[0, :] = 0

        # define the evaluation rewards
        val_rewards = np.empty((num_episodes + 1, len(policies), num_models))
        val_mean = np.empty((num_episodes + 1, len(policies)))
        val_var = np.empty((num_episodes + 1, len(policies)))

        # set value for first episode
        val_rewards[0, :, :] = 0
        val_mean[0, :] = 0
        val_var[0, :] = 0

        # get the learn operations

        # retrieve the learn operations
        update_and_receive_rewards = [agent.q_tensor_update for agent in agents]
        perform_ops = [agent.perform_operation for agent in agents]

        reset_ops = [envs.reset_op for envs in environments]
        cum_rew_ops = [envs.cum_rewards for envs in environments]

        # iterate over episodes
        for episode in range(1, num_episodes + 1):

            print("Current Episode is: " + str(episode))

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

        # Create Q Value Function plots
        approx_density = [agent.all_densities_model for agent in agents]
        reference_density = [agent.all_densities for agent in agents]

        feed_dict = {}
        for agent in agents:
            feed_dict[agent.use_best] = True

        approx_density_res, reference_density_res = sess.run([approx_density, reference_density], feed_dict)

# Create first plot
fig_error = plt.figure(0)
top = fig_error.add_subplot(211)
bottom = fig_error.add_subplot(212)
top.set_title("Training Error")
bottom.set_title("Validation Error")
top.axhline(y=1, color='r', linestyle=':', label='Optimal')
bottom.axhline(y=1, color='r', linestyle=':', label='Optimal')

# plot using the correct colors
for i in range(np.size(training_mean, axis=1)):
    handles_top = top.plot(training_mean[:, i], color=color_pool[i][0], label=policies[i][0])
    handles_bottom = bottom.plot(val_mean[:, i], color=color_pool[i][0], label=policies[i][0])

top.legend()
bottom.legend()

# Create the heat plot
num_display_models = 3
fig_heatmap = plt.figure(1)
approx_plots = [None] * num_display_models
real_plots = [None] * num_display_models
num = 100 * num_display_models + 20

for i in range(num_display_models):
    approx_plots[i] = fig_heatmap.add_subplot(num + 2 * i + 1)
    real_plots[i] = fig_heatmap.add_subplot(num + 2 * i + 2)

for i in range(num_display_models):
    approx_plots[i].imshow(np.transpose(approx_density_res[0][i, 0, :, :]), interpolation='nearest')
    real_plots[i].imshow(np.transpose(reference_density_res[0][i, 0, :, :]), interpolation='nearest')

plt.show()
