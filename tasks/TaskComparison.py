import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

from agents.QLearningAgent import QLearningAgent
from collection.ColorCollection import ColorCollection
from collection.PolicyCollection import PolicyCollection
from environments.GridWorld import GridWorld
from environments.BinaryFlipEnvironment import BinaryFlipEnvironment
from environments.DeepSeaExploration import DeepSeaExploration
from environments.DeepSeaExplorationTwo import DeepSeaExplorationTwo
from environments.DeepSeaExplorationThree import DeepSeaExplorationThree
from environments.DeepSeaExplorationFour import DeepSeaExplorationFour
from environments.ExplorationChain import ExplorationChain
from manager.DirectoryManager import DirectoryManager
from plots.MultiDimensionalHeatMap import MultiDimensionalHeatmap

# ------------------------------ SETTINGS ------------------------------------

run = list()

new_envs = [[BinaryFlipEnvironment, [4], lambda n: 8 * n, 250]]
new_batch_names = [['bootstrapped', []], ['cb_pseudo_count', []],
                   ['eps_greedy', []], ['boltzmann', []], ['ucb', []],
                   ['optimistic', []], ['ucb', []],
                   ['bootstrapped_heads_per_sample', []], ['ucb_infogain', []],
                   ['pc_pseudo_count', []], ['deterministic_bootstrapped_cb_pseudo_count', []]]

run.append([new_envs, new_batch_names])
run.append([new_envs, new_batch_names])

save_directory = "run/CompleteTestRun"
#num_models = 1000
num_episodes = 3000

#record_indices = []  # 0, 1, 2, 3]
plot_models = 1
plot_heads = 5
save_frame = 1
fps = 15
shared_learning_steps = 32

for [all_envs, batch_names] in run:
    for [env_build, problem_sizes, problem_to_step, num_models] in all_envs:

        for N in problem_sizes:
            for [batch_name, record_indices] in batch_names:

                # create variable for the steps a30nd do this amount of steps.
                seed = 3
                num_steps = problem_to_step(N)

                # ------------------------------- SCRIPT -------------------------------------

                tf.set_random_seed(seed)
                graph = tf.Graph()
                with graph.as_default():
                    tf_config = tf.ConfigProto(log_device_placement=True)
                    tf_config.gpu_options.allow_growth=True
                    with tf.Session(graph=graph, config=tf_config) as sess:

                        env = env_build("test", [num_models], N)
                        state_space = env.state_space
                        action_space = env.action_space
                        log_action_size = action_space.get_log2_size()

                        time_frame = 20
                        color_pool = ColorCollection.get_colors()

                        # define the different policies you want to try out
                        dir_manager = DirectoryManager(save_directory)
                        dir_manager.set_env_dir("{}_{}".format(env.get_name(), N))
                        dir_manager.set_agent_dir(batch_name)
                        policies = PolicyCollection.get_batch(batch_name)

                        # --------------------- Determine the optimal reward --------------------

                        # Determine the agent count
                        num_policies = len(policies)
                        optimal_ih_rew, minimal_ih_rew, min_q, max_q, _ = env.get_optimal(num_steps, 0.99)

                        # --------------------------------------------------------------------------

                        # Iterate over all policies and create an agent using that specific policy
                        agents = list()
                        q_plots = list()
                        density_plots = list()
                        environments = list()
                        densities = list()
                        q_functions = list()
                        get_best_shared = list()
                        for pol_num in range(num_policies):

                            # Get policies and unique name
                            pe = policies[pol_num]
                            unique_name = str(pol_num)

                            # extract important fields
                            policy = pe[1]
                            policy_config = pe[2]
                            policy_config['num_models'] = num_models
                            policy_config['min_q'] = min_q
                            policy_config['max_q'] = max_q
                            policy_config['action_space'] = action_space

                            current_env = env.clone(unique_name)
                            environments.append(current_env)
                            agent = QLearningAgent(sess, unique_name, current_env, policy, policy_config)
                            agents.append(agent)

                            if plot_models > 0 and pol_num in record_indices:

                                # setup densities
                                if 'pseudo_count_type' in policy_config and policy_config['pseudo_count_type']:
                                    num_densities = 2
                                    densities.append([agent.cb_complete_densities, agent.ref_complete_densities])
                                else:
                                    num_densities = 1
                                    densities.append([agent.ref_complete_densities])

                                # setup q functions
                                q_functions.append([agent.q_tensor])

                                # get the learn operations
                                q_plots.append(
                                    MultiDimensionalHeatmap("q_func_{}".format(pol_num), 1,
                                                            [plot_models, np.minimum(policy_config['num_heads'], plot_heads),
                                                             state_space.get_size(), action_space.get_size()],
                                                            0.8, 'viridis'))

                                density_plots.append(
                                    MultiDimensionalHeatmap("density_{}".format(pol_num), num_densities,
                                                            [plot_models, np.minimum(policy_config['num_heads'], plot_heads),
                                                             state_space.get_size(), action_space.get_size()],
                                                            0.8, 'inferno'))

                                if 'shared_learning' in policy_config and policy_config['shared_learning']:
                                    get_best_shared.append(agent.get_best_heads)

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
                        val_rewards = np.zeros((num_episodes + 1, len(policies), num_models))
                        val_mean = np.zeros((num_episodes + 1, len(policies)))
                        val_var = np.zeros((num_episodes + 1, len(policies)))

                        # set value for first episode
                        val_rewards[0, :, :] = 0
                        val_mean[0, :] = 0
                        val_var[0, :] = 0

                        feed_dict = {}
                        for agent in agents:
                            feed_dict[agent.use_best] = True

                        # retrieve the learn operations
                        update_and_receive_rewards = [agent.q_tensor_update for agent in agents]
                        perform_ops = [agent.apply_actions for agent in agents]

                        reset_ops = [envs.reset_op for envs in environments]
                        cum_rew_ops = [envs.cum_rewards for envs in environments]

                        # start the recording
                        for i in range(len(q_plots)):
                            q_plots[i].start_recording(dir_manager.agent_root, fps)
                            density_plots[i].start_recording(dir_manager.agent_root, fps)

                        # iterate over episodes
                        for episode in range(1, num_episodes + 1):
                            start = time.time()

                            # reset all environments
                            sess.run(reset_ops)

                            # for each agent sample a new head
                            state_dict = {}
                            for k in range(num_policies):
                                agents[k].sample_head()
                                state_dict[agents[k].use_best] = False

                            # repeat this for the number of steps
                            for k in range(num_steps):

                                if len(get_best_shared) > 0 and k % shared_learning_steps == 0:
                                    sess.rún(shared_learning_steps)

                                # receive rewards and add
                                sess.run(update_and_receive_rewards, feed_dict=state_dict)

                            # copy values
                            training_rewards[episode, :, :] = (sess.run(cum_rew_ops) - minimal_ih_rew) / (optimal_ih_rew - minimal_ih_rew)

                            # determine mean and variance
                            training_mean[episode, :] = np.mean(training_rewards[episode, :, :], axis=1)
                            training_var[episode, :] = np.var(training_rewards[episode, :, :], axis=1)

                            if False:

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

                            # when a frame should be recorded
                            if len(record_indices) > 0 and (episode - 1) % save_frame == 0:

                                feed_dict = {}
                                for agent in agents:
                                    feed_dict[agent.use_best] = True

                                res_q_functions, res_densities = sess.run([q_functions, densities], feed_dict)
                                for i in range(len(record_indices)):

                                    # store the q plot
                                    q_plots[i].plot(res_q_functions[i])
                                    q_plots[i].store_frame()

                                    # store the density
                                    density_plots[i].plot(res_densities[i])
                                    density_plots[i].store_frame()

                            print("\tEpisode {} finished after {} ms".format(episode, round((time.time() - start) * 1000, 2)))

                # start the recording
                for i in range(len(q_plots)):
                    q_plots[i].stop_recording()
                    density_plots[i].stop_recording()

                # --------------------------------------------
                # save the plots with all errors
                # --------------------------------------------

                dir_manager.save_tr_va_plots(training_mean, val_mean, [policy[0] for policy in policies], "all_policies.eps")

                # of course print the best policy with variance
                cum_mean = np.sum(training_mean, axis=0)
                best_policy = np.argmax(cum_mean)
                dir_manager.save_tr_va_plots(training_mean[:, best_policy:best_policy+1], val_mean[:, best_policy:best_policy+1], policies[best_policy][0], "best_policy.eps")

                # --------------------------------------------
                # Store the rewards etc.
                # --------------------------------------------

                agent_root = dir_manager.agent_root
                np.savetxt(os.path.join(agent_root, "tr_rewards_mean.np"), training_mean)
                np.savetxt(os.path.join(agent_root, "tr_rewards_var.np"), training_var)
                np.savetxt(os.path.join(agent_root, "va_rewards_mean.np"), val_mean)
                np.savetxt(os.path.join(agent_root, "va_rewards_var.np"), val_var)