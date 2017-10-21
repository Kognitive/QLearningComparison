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
from environments.ExplorationTree import ExplorationTree
from environments.SharedLearningChain import SharedLearningChain
from manager.DirectoryManager import DirectoryManager
from plots.MultiDimensionalHeatMap import MultiDimensionalHeatmap

# ------------------------------ SETTINGS ------------------------------------

run = list()
new_batch_names = [['eps_greedy', []], ['shared_bootstrap', []], ['bootstrapped', []],
                    ['boltzmann', []], ['cb_pseudo_count', []],
                    ['optimistic', []], ['ucb', []],
                    ['bootstrapped_heads_per_sample', []], ['ucb_infogain', []], ['deterministic_bootstrapped_cb_pseudo_count', []]]
fb = 2
seed = fb
if fb == 0:
    new_batch_names = [['optimistic', []], ['ucb', []], ['boltzmann', []], ['bootstrapped', []], ['cb_pseudo_count', []],
                    ['bootstrapped_heads_per_sample', []], ['ucb_infogain', []], ['deterministic_bootstrapped_cb_pseudo_count', []]]
    new_envs = [[[ExplorationChain], [200], lambda n: n + 9, 1500, "exp_chain"]]
    run.append([new_envs, new_batch_names])

elif fb == 1:

    new_batch_names = [['optimistic', []], ['ucb', []], ['cb_pseudo_count', []],
                    ['bootstrapped_heads_per_sample', []], ['ucb_infogain', []], ['deterministic_bootstrapped_cb_pseudo_count', []]]
    new_envs = [[[SharedLearningChain], [133], lambda n: n, 1500, "shared_chain"]]
    run.append([new_envs, new_batch_names])

elif fb == 2:

    new_batch_names = [['deterministic_bootstrapped_cb_pseudo_count', []]]
    new_envs = [[[DeepSeaExplorationTwo], [19], lambda n: n, 750, "deep_sea_two"]]
    run.append([new_envs, new_batch_names])

    new_batch_names = [['ucb_infogain', []],
                       ['deterministic_bootstrapped_cb_pseudo_count', []]]
    new_envs = [[[BinaryFlipEnvironment], [6], lambda n: 8 * n, 1500, "bin_flip"]]
    run.append([new_envs, new_batch_names])

    # new_envs = [[BinaryFlipEnvironment, [6], lambda n: n ** 2, 2500]]
    # new_batch_names = [['eps_greedy', []], ['shared_bootstrap', []], ['bootstrapped', []],
    #                    ['boltzmann', []], ['cb_pseudo_count', []],
    #                    ['optimistic', []], ['ucb', []],
    #                    ['bootstrapped_heads_per_sample', []], ['ucb_infogain', []],
    #                    ['pc_pseudo_count', []], ['deterministic_bootstrapped_cb_pseudo_count',[]]]
    # run.append([new_envs, new_batch_names])
    #
    # new_envs = [[GridWorld, [10], lambda n: 2 * n, 2500]]
    # new_batch_names = [['eps_greedy', []], ['bootstrapped_heads_per_sample', []], ['ucb_infogain', []]]
    # run.append([new_envs, new_batch_names])
    #
    # new_envs = [[DeepSeaExploration, [20], lambda n: n, 625]]
    # new_batch_names = [['eps_greedy', []], ['shared_bootstrap', []], ['bootstrapped', []],
    #                    ['boltzmann', []], ['cb_pseudo_count', []],
    #                    ['optimistic', []], ['ucb', []],
    #                    ['bootstrapped_heads_per_sample', []], ['ucb_infogain', []],
    #                    ['pc_pseudo_count', []], ['deterministic_bootstrapped_cb_pseudo_count',[]]]
    # run.append([new_envs, new_batch_names])
    #
    # new_envs = [[DeepSeaExplorationTwo, [20], lambda n: n, 625]]
    # new_batch_names = [['eps_greedy', []], ['shared_bootstrap', []], ['bootstrapped', []],
    #                    ['boltzmann', []], ['cb_pseudo_count', []],
    #                    ['optimistic', []], ['ucb', []],
    #                    ['bootstrapped_heads_per_sample', []], ['ucb_infogain', []],
    #                    ['pc_pseudo_count', []], ['deterministic_bootstrapped_cb_pseudo_count',[]]]
    # run.append([new_envs, new_batch_names])
    #
    # new_envs = [[DeepSeaExplorationThree, [20], lambda n: n, 625]]
    # new_batch_names = [['eps_greedy', []], ['shared_bootstrap', []], ['bootstrapped', []],
    #                    ['boltzmann', []], ['cb_pseudo_count', []],
    #                    ['optimistic', []], ['ucb', []],
    #                    ['bootstrapped_heads_per_sample', []], ['ucb_infogain', []],
    #                    ['pc_pseudo_count', []], ['deterministic_bootstrapped_cb_pseudo_count',[]]]
    # run.append([new_envs, new_batch_names])
    #
    # new_envs = [[DeepSeaExplorationFour, [20], lambda n: n, 625]]
    # new_batch_names = [['eps_greedy', []], ['shared_bootstrap', []], ['bootstrapped', []],
    #                    ['boltzmann', []], ['cb_pseudo_count', []],
    #                    ['optimistic', []], ['ucb', []],
    #                    ['bootstrapped_heads_per_sample', []], ['ucb_infogain', []],
    #                    ['pc_pseudo_count', []], ['deterministic_bootstrapped_cb_pseudo_count',[]]]
    # run.append([new_envs, new_batch_names])

save_directory = "run/RunBiggerProblem"
#num_models = 1000
num_episodes = 7000
#record_indices = []  # 0, 1, 2, 3]
plot_models = 1
plot_heads = 5
save_frame = 1
fps = 15

for [all_envs, batch_names] in run:
    for [env_build, problem_sizes, problem_to_step, num_models, env_name] in all_envs:
        for N in problem_sizes:
            for [batch_name, record_indices] in batch_names:

                # define the different policies you want to try out
                dir_manager = DirectoryManager(save_directory, "{}_{}".format(env_name, N), batch_name)

                # get policy collection
                policies = PolicyCollection.get_batch(batch_name)

                # define the evaluation rewards
                m = len(env_build)
                training_rewards = np.empty((num_episodes + 1, m, len(policies), num_models))
                training_mean = np.empty((num_episodes + 1, m, len(policies)))
                training_var = np.empty((num_episodes + 1, m, len(policies)))

                # set value for first episode
                training_rewards[0, :, :, :] = 0
                training_mean[0, :, :] = 0
                training_var[0, :, :] = 0

                min_rew = 10000000
                max_rew = -min_rew

                for bi in range(len(env_build)):
                    build_env = env_build[bi]

                    # --------
                    # create variable for the steps a30nd do this amount of steps.
                    num_steps = problem_to_step(N)

                    tf.set_random_seed(seed)
                    graph = tf.Graph()
                    with graph.as_default():

                        tf_config = tf.ConfigProto(log_device_placement=True)
                        tf_config.intra_op_parallelism_threads = 8
                        tf_config.inter_op_parallelism_threads = 8
                        tf_config.gpu_options.allow_growth=True

                        with tf.Session(graph=graph, config=tf_config) as sess:

                            env = build_env("test", [num_models], N)
                            state_space = env.state_space
                            action_space = env.action_space
                            log_action_size = action_space.get_log2_size()

                            time_frame = 20
                            color_pool = ColorCollection.get_colors()

                            # --------------------- Determine the optimal reward --------------------

                            # Determine the agent count
                            num_policies = len(policies)
                            optimal_ih_rew, minimal_ih_rew, min_q, max_q, _ = env.get_optimal(num_steps, 0.99)
                            min_rew = np.minimum(minimal_ih_rew, min_rew)
                            max_rew = np.maximum(optimal_ih_rew, max_rew)

                            # --------------------------------------------------------------------------

                            # Iterate over all policies and create an agent using that specific policy
                            agents = list()
                            q_plots = list()
                            density_plots = list()
                            environments = list()
                            densities = list()
                            q_functions = list()
                            get_best_shared = list()
                            shared_steps = list()
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
                                if 'shared_learning' in policy_config:
                                    shared_steps.append(policy_config['shared_steps'])

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
                                    shd_stp_lst = list()
                                    for m in range(num_policies):
                                        if 'shared_steps' in policies[m][2]:
                                            if shared_steps[m] > 0 and (k + (episode * num_steps)) % shared_steps[m] == 0:
                                                shd_stp_lst.append(agents[m].get_best_heads)

                                    # receive rewards and add
                                    sess.run(update_and_receive_rewards, feed_dict=state_dict)
                                    sess.run(shd_stp_lst, feed_dict=state_dict)

                                # copy values
                                training_rewards[episode, bi, :, :] = (sess.run(cum_rew_ops) - minimal_ih_rew) / (optimal_ih_rew - minimal_ih_rew)

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
                # determine mean and variance
                training_mean = np.mean(training_rewards, axis=(1,3))
                training_var = np.var(training_rewards, axis=(1,3))

                dir_manager.save_tr_va_plots(training_mean, None, [policy[0] for policy in policies], "all_policies.eps")

                # of course print the best policy with variance
                ##cum_mean = np.sum(training_mean, axis=0)
                #best_policy = np.argmax(cum_mean)
                #dir_manager.save_tr_va_plots(training_mean[:, best_policy:best_policy+1], None, policies[best_policy][0], "best_policy.eps")

                # --------------------------------------------
                # Store the rewards etc.
                # --------------------------------------------

                agent_root = dir_manager.root
                np.savetxt(os.path.join(agent_root, "tr_rewards_mean.np"), training_mean)
                np.savetxt(os.path.join(agent_root, "tr_rewards_var.np"), training_var)