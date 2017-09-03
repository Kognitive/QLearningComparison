import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from agents.QLearningAgent import QLearningAgent
from policies.GreedyPolicy import GreedyPolicy
from policies.BoltzmannPolicy import BoltzmannPolicy
from policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy
from environments.ExplorationChain import ExplorationChain
from environments.GridWorld import GridWorld

# the number of models an average should be taken

# ------------------------------ SETTINGS ------------------------------------
N = 20
env_build = ExplorationChain

# create variable for the steps and do this amount of steps.
num_models = 10000
show_models = 3
num_episodes = 100
num_steps = 2 * N

# create
new_env = lambda: env_build(num_models, N)
env = new_env()
state_space = env.get_state_space()
action_space = env.get_action_space()
log_action_size = action_space.get_log2_size()

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

time_frame = 20
color_pool = [
    ['#1455bc', '#81b0f945'],
    ['#e58d09', '#edc89045'],
    ['#ad0bbc', '#bb5ec445'],
    ['#56b21c', '#b4db9b45'],
    ['#870404', '#d1555545'],
]

# define the different policies you want to try out
policies = [
     #['$\epsilon$-Greedy (0.001)', EpsilonGreedyPolicy, {'action_space': action_space, 'epsilon': 0.001}],
     #['$\epsilon$-Greedy (0.005)', EpsilonGreedyPolicy, {'action_space': action_space, 'epsilon': 0.005}],
     #['$\epsilon$-Greedy (0.01)', EpsilonGreedyPolicy, {'action_space': action_space, 'epsilon': 0.01}],
     #['$\epsilon$-Greedy (0.05)', EpsilonGreedyPolicy, {'action_space': action_space, 'epsilon': 0.05}],
     #['$\epsilon$-Greedy (0.1)', EpsilonGreedyPolicy, {'action_space': action_space, 'epsilon': 0.1}],
     #['$\epsilon$-Greedy (0.2)', EpsilonGreedyPolicy, {'action_space': action_space, 'epsilon': 0.2}],
     #['$\epsilon$-Greedy (0.3)', EpsilonGreedyPolicy, {'action_space': action_space, 'epsilon': 0.3}],
    #['Boltzmann (0.1)', BoltzmannPolicy, {'temperature': 0.1}],
    #['Boltzmann (1)', BoltzmannPolicy, {'temperature': 1}],
    #['Boltzmann (10)', BoltzmannPolicy, {'temperature': 10}]
    ['Deterministic Bootstrapped (3)', GreedyPolicy, {'action_space': action_space, 'num_heads': 3}],
    ['Deterministic Bootstrapped (7)', GreedyPolicy, {'action_space': action_space, 'num_heads': 7}],
    ['Deterministic Bootstrapped (12)', GreedyPolicy, {'action_space': action_space, 'num_heads': 12}],
    ['Deterministic Bootstrapped (2)', GreedyPolicy, {'action_space': action_space, 'num_heads': 12}]
    # ['$\epsilon$-Greedy Bootstrapped', EpsilonGreedyPolicy, {'action_space': action_space, 'epsilon': 0.1, 'num_heads': 3}],
    #['$\epsilon$-Greedy Bootstrapped (0.05, 3)', EpsilonGreedyPolicy,
    # {'action_space': action_space, 'epsilon': 0.05, 'num_heads': 7}],
    #['$\epsilon$-Greedy Bootstrapped (0.1, 3)', EpsilonGreedyPolicy,
    # {'action_space': action_space, 'epsilon': 0.1, 'num_heads': 3}],
    #['$\epsilon$-Greedy Bootstrapped (0.05, 5)', EpsilonGreedyPolicy,
    # {'action_space': action_space, 'epsilon': 0.05, 'num_heads': 20}],
    #['$\epsilon$-Greedy Bootstrapped (0.1, 5)', EpsilonGreedyPolicy,
    # {'action_space': action_space, 'epsilon': 0.1, 'num_heads': 20}]
    # ['#56b21c', '#b4db9b45', 'Boltzmann Bootstrapped', BoltzmannPolicy, {'temperature': 3, 'num_heads': 5}]
    # ['#ad0bbc', '#bb5ec445', 'Optimistic-Greedy', GreedyPolicy, {'optimistic': True}],
    # ['#56b21c', '#b4db9b45', 'UCB1-Greedy', EpsilonGreedyPolicy, {'ucb': True, 'p': 2.0, 'action_space': action_space, 'epsilon': 0.3}],
    # ['#870404', '#d1555545', 'Pseudo-Count-Greedy 1', GreedyPolicy, {'pseudo_count': True, 'beta': 1000}]
]

# --------------------- Determine the optimal reward --------------------

# Determine the agent count
policy_count = len(policies)
optimal_ih_rew = env.get_optimal(num_steps, 0.99)

# --------------------------------------------------------------------------

graph = tf.Graph()
with graph.as_default():
    with tf.Session(graph=graph) as sess:

        # Iterate over all policies and create an agent using that specific policy
        agents = list()
        for pe_num in range(policy_count):

            # Get polic and unqiue name
            pe = policies[pe_num]
            unique_name = str(pe_num)

            # extract important fields
            policy_name = pe[1]
            policy_config = pe[2]
            policy_config['num_models'] = num_models

            agents.append(QLearningAgent(sess, unique_name, state_space, action_space, policy_name, policy_config))

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

        # create one environment per agent
        environments = [new_env() for agent in agents]

        # get the choose operations
        choose_operations = [agent.normal_actions for agent in agents]

        # get the choose operations
        choose_best_operations = [agent.best_actions for agent in agents]

        # get the learn operations
        learn_operations = [agent.q_tensor_update for agent in agents]

        # iterate over episodes
        for episode in range(1, num_episodes + 1):

            print("Current Episode is: " + str(episode))

            # reset all environments
            [environments[i].reset() for i in range(policy_count)]
            episode_rewards = np.zeros((policy_count, num_models), dtype=np.float64)

            # for each agent sample a new head
            for k in range(policy_count):
                agents[k].sample_head()

            # repeat this for the number of steps
            for k in range(num_steps):

                # Collect the current state of each environment in a dict
                state_dict = {}
                for i in range(policy_count):
                    state_dict[agents[i].current_states] = environments[i].get_states()

                # Execute all actions and collect rewards
                actions = sess.run(choose_operations, feed_dict=state_dict)
                rewards = np.array([environments[i].actions(actions[i]) for i in range(policy_count)])
                episode_rewards += rewards

                # fill the other details for each agent
                for i in range(policy_count):
                    state_dict[agents[i].lr] = [1]
                    state_dict[agents[i].discount] = [0.99]
                    state_dict[agents[i].rewards] = rewards[i]
                    state_dict[agents[i].actions] = actions[i]
                    state_dict[agents[i].next_states] = environments[i].get_states()

                # learn all tuples
                sess.run(learn_operations, feed_dict=state_dict)

            # copy values
            training_rewards[episode, :, :] = episode_rewards

            # determine mean and variance
            training_mean[episode, :] = np.mean(training_rewards[episode, :, :], axis=1)
            training_var[episode, :] = np.var(training_rewards[episode, :, :], axis=1)

            # reset all environments
            [environments[i].reset() for i in range(policy_count)]
            episode_rewards = np.zeros((policy_count, num_models), dtype=np.float64)

            # for each agent sample a new head
            for k in range(policy_count):
                agents[k].sample_head()

            # repeat this for the number of steps
            for k in range(num_steps):

                # Collect the current state of each environment in a dict
                state_dict = {}
                for i in range(policy_count):
                    state_dict[agents[i].current_states] = environments[i].get_states()

                # Execute all actions and collect rewards
                actions = sess.run(choose_best_operations, feed_dict=state_dict)
                rewards = np.array([environments[i].actions(actions[i]) for i in range(policy_count)])
                episode_rewards += rewards

            # copy values
            val_rewards[episode, :, :] = episode_rewards

            # determine mean and variance
            val_mean[episode, :] = np.mean(val_rewards[episode, :, :], axis=1)
            val_var[episode, :] = np.var(val_rewards[episode, :, :], axis=1)

fig_error = plt.figure(0)
top = fig_error.add_subplot(211)
bottom = fig_error.add_subplot(212)
handles_top = top.plot(training_mean)
handles_bottom = bottom.plot(val_mean)
label_list = [policies[i][0] for i in range(policy_count)]
top.legend(handles_top, label_list)
bottom.legend(handles_bottom, label_list)
top.axhline(y=optimal_ih_rew, color='r', linestyle=':', label='Optimal')
bottom.axhline(y=optimal_ih_rew, color='r', linestyle=':', label='Optimal')
plt.show()