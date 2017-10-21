import numpy as np
import tensorflow as tf
import time
import os

import multiprocessing as mp
from multiprocessing import Queue
from manager.DirectoryManager import DirectoryManager
from agents.QLearningAgent import QLearningAgent
from collection.PolicyCollection import PolicyCollection

from environments.DeepSeaExploration import DeepSeaExploration
from environments.DeepSeaExplorationTwo import DeepSeaExplorationTwo
from environments.DeepSeaExplorationThree import DeepSeaExplorationThree
from environments.DeepSeaExplorationFour import DeepSeaExplorationFour


class MultithreadedRunner:

    def __init__(self, directory, num_threads=16):
        self.executor = mp.Pool()
        self.configs = list()
        self.merges = list()
        self.dir = directory

    def start(self):
        self.executor.map(self.run_config, self.configs)
        self.executor.close()
        self.executor.join()

        for [folder, merge_lst, size, policy] in self.merges:
            res = [results[el] for el in merge_lst]
            stacked_rewards = np.concatenate(res, axis=2)

            p = os.path.join(self.dir, "{}{}".format(folder, size), policy)
            dirman = DirectoryManager(self.dir, "{}{}".format(folder, size), policy)

            pm = os.path.join(p, "tr_rewards.np")
            np.savetxt(stacked_rewards, pm)

    def add_task(self, name, envs, problem_size, num_models, num_episodes, prob_size_to_step, policies):
        for policy in policies:
            merge_entry = list()
            for env in envs:
                merge_entry.append(len(self.configs))
                self.configs.append([env, problem_size, num_models, num_episodes, prob_size_to_step(problem_size), policy])

            self.merges.append([name, merge_entry, prob_size_to_step(problem_size), policy])

    def run_config(self, configuration):
        env, problem_size, num_models, num_episodes, num_steps, policy = configuration
        policies = PolicyCollection.get_batch(policy)

        # create a new graph
        graph = tf.Graph()
        with graph.as_default():

            # and a configuration as well.
            tf_config = tf.ConfigProto(log_device_placement=True)
            tf_config.intra_op_parallelism_threads = 8
            tf_config.inter_op_parallelism_threads = 8
            tf_config.gpu_options.allow_growth = True

            with tf.Session(graph=graph, config=tf_config) as sess:

                env = env("test", [num_models], problem_size)
                state_space = env.state_space
                action_space = env.action_space

                # --------------------- Determine the optimal reward --------------------

                # Determine the agent count
                num_policies = len(policies)
                optimal_ih_rew, minimal_ih_rew, min_q, max_q, _ = env.get_optimal(num_steps, 0.99)

                # --------------------------------------------------------------------------

                # Iterate over all policies and create an agent using that specific policy
                agents = list()
                environments = list()
                densities = list()
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
                    shared_steps.append(policy_config['shared_steps'])
                    densities.append([agent.ref_complete_densities])

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
                reset_ops = [envs.reset_op for envs in environments]
                cum_rew_ops = [envs.cum_rewards for envs in environments]

                # create trainings rewards
                tr_rewards = np.zeros((num_episodes, num_policies, num_models))

                # iterate over episodes
                for episode in range(num_episodes):

                    # reset all environments
                    sess.run(reset_ops)

                    # for each agent sample a new head
                    state_dict = {}
                    for k in range(num_policies):
                        agents[k].sample_head()
                        state_dict[agents[k].use_best] = False

                    # repeat this for the number of steps
                    for k in range(num_steps):
                        for m in range(num_policies):
                            if shared_steps[k] > 0 and k % shared_steps[k] == 0:
                                sess.run(agents[k].get_best_heads)

                        # receive rewards and add
                        sess.run(update_and_receive_rewards, feed_dict=state_dict)

                    # copy values
                    tr_rewards[episode, :, :] = sess.run(cum_rew_ops)

        return tr_rewards


x = MultithreadedRunner("run/Draft2500DeepSea")
policies = ['bootstrapped', 'cb_pseudo_count',
            'eps_greedy', 'boltzmann', 'ucb', 'optimistic',
            'bootstrapped_heads_per_sample', 'ucb_infogain',
            'pc_pseudo_count', 'deterministic_bootstrapped_cb_pseudo_count']

x.add_task("deep_sea", [DeepSeaExploration, DeepSeaExplorationTwo, DeepSeaExplorationThree, DeepSeaExplorationFour],
           10, 250, 7000, lambda n: n, policies)

x.start()