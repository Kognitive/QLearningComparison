import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.animation as manimation
import numpy as np
import tensorflow as tf
import os
import datetime

from agents.QLearningAgent import QLearningAgent
from policies.GreedyPolicy import GreedyPolicy
from policies.BoltzmannPolicy import BoltzmannPolicy
from policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy
from environments.ExplorationChain import ExplorationChain
from environments.DeepSeaExploration import DeepSeaExploration
from environments.GridWorld import GridWorld
from environments.BinaryFlipEnvironment import BinaryFlipEnvironment
from plots.MultiDimensionalHeatMap import MultiDimensionalHeatmap

# the number of models an average should be taken

# ------------------------------ SETTINGS ------------------------------------
N = 5
env_build = BinaryFlipEnvironment
mpl.verbose.set_level("helpful")

# create variable for the steps and do this amount of steps.
num_models = 30
show_models = 3
num_episodes = 10
num_steps = 8 * N

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
            ['Deterministic Bootstrapped', GreedyPolicy, {'action_space': action_space, 'num_heads': 12, 'heads_per_sample': 3}]]
        # --------------------- Determine the optimal reward --------------------

        # Determine the agent count
        num_policies = len(policies)
        optimal_ih_rew, min_q, max_q = env.get_optimal(num_steps, 0.99)

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
            policy_config['min_q'] = min_q
            policy_config['max_q'] = max_q
            policy_config['action_space'] = action_space

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
        heatmap = MultiDimensionalHeatmap("agent_1", "Agent", 1, [5, policies[0][2]['num_heads'], state_space.get_size(), action_space.get_size()], 0.8, 'viridis')

        feed_dict = {}
        for agent in agents:
            feed_dict[agent.use_best] = True

        approx_density = [[agent.cb_complete_densities, agent.ref_complete_densities] for agent in agents]
        q_functions = [[agent.q_tensor] for agent in agents]

        # retrieve the learn operations
        update_and_receive_rewards = [agent.q_tensor_update for agent in agents]
        perform_ops = [agent.apply_actions for agent in agents]

        reset_ops = [envs.reset_op for envs in environments]
        cum_rew_ops = [envs.cum_rewards for envs in environments]

        timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
        heatmap.start_recording('../movies/{}'.format(timestamp))

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

            ensity_res = sess.run(q_functions, feed_dict)
            heatmap.plot(ensity_res[0])
            heatmap.writer.grab_frame()

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
        heatmap.save_recording()

        if approx_density[0][0] == approx_density[0][1]:
            approx_density = [[a] for [a, b] in approx_density]

        feed_dict = {}
        for agent in agents:
            feed_dict[agent.use_best] = True

        density_res = sess.run(approx_density, feed_dict)

# Create first plot
fig_error = plt.figure("Error")
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

for i in range(len(agents)):
    heatmap = MultiDimensionalHeatmap(len(density_res[i]), "Agent", i, [5, policies[i][2]['num_heads'], state_space.get_size(), action_space.get_size()], 0.8, 'inferno')
    heatmap.plot(density_res[i])

plt.show(block = True)