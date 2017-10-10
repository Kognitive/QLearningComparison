# MIT License
#
# Copyright (c) 2017 Markus Semmler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from policies.GreedyPolicy import GreedyPolicy
from policies.BoltzmannPolicy import BoltzmannPolicy
from policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy


class PolicyCollection:
    """Represents a policy collection can be used to organize the policy batches."""

    # create a new dictionary
    policy_def = dict()
    init = False

    @staticmethod
    def initialize():

        # simple deterministic bootstrapped agent
        PolicyCollection.register("bootstrapped",
                                  "Deterministic Bootstrapped (K={})",
                                  GreedyPolicy,
                                  [['num_heads', [1, 3, 5, 7, 10]], ['init_gaussian', [False]]])

        PolicyCollection.register("eps_greedy",
                                  "$\\epsilon$-Greedy ($\\epsilon$={})",
                                  EpsilonGreedyPolicy,
                                  [['epsilon', [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]]])

        PolicyCollection.register("boltzmann",
                                  "Boltzmann ($\\beta$={})",
                                  BoltzmannPolicy,
                                  [['temperature', [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]]])

        PolicyCollection.register("cb_pseudo_count",
                                  "CB Pseudo Count ($\\beta$={})",
                                  GreedyPolicy,
                                  [['beta', [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 1, 5, 10]],
                                   ['pseudo_count', [True]], ['pseudo_count_type', ['count_based']], ['create_density', [True]]])

        PolicyCollection.register("optimistic",
                                  "Optimistic",
                                  GreedyPolicy,
                                  [['optimistic', [True]]])

        PolicyCollection.register("ucb",
                                  "UCB-Greedy (p={})",
                                  GreedyPolicy,
                                  [['p', [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 1, 5, 10]], ['ucb', [True]], ['create_density', [True]]])

        PolicyCollection.register("bootstrapped_heads_per_sample",
                                  "Deterministic Bootstrapped (H={}, K={})",
                                  GreedyPolicy,
                                  [['heads_per_sample', [1, 2, 3, 5, 7]], ['num_heads', [7]], ['init_gaussian', [False]]])

        PolicyCollection.register("pc_pseudo_count",
                                  "PC Pseudo Count (lr={}, \\beta={})",
                                  GreedyPolicy,
                                  [['learning_rate', [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]], ['beta', [0.05]],
                                   ['pseudo_count', [True]], ['pseudo_count_type', ['pseudo_count']]])

        PolicyCollection.register("deterministic_bootstrapped_cb_pseudo_count",
                                  "Deterministic PC Bootstrapped (K={},  \\beta={})",
                                  GreedyPolicy,
                                  [['num_heads', [7]], ['beta', [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 1]],
                                   ['pseudo_count', [True]], ['pseudo_count_type', ['count_based']], ['create_density', [True]], ['init_gaussian', [False]]]),

        PolicyCollection.register("ucb_infogain",
                                  "UCB-InfoGain (lambda={})",
                                  GreedyPolicy,
                                  [['lambda', [0.01, 0.05, 0.1, 0.5]], ['ucb_infogain', [True]], ['num_heads', [7]], ['info_gain_temp', [0.01, 0.05, 0.1, 0.5, 1]]])

    @staticmethod
    def register(key, name, policy, parameters):
        PolicyCollection.policy_def[key] = [name, policy, parameters]

    @staticmethod
    def get_batch(key):
        """Should deliver a batch of policies to test."""

        if not PolicyCollection.init:
            PolicyCollection.init = True
            PolicyCollection.initialize()

        if key not in PolicyCollection.policy_def:
            raise NotImplementedError("You have to define the batch parameters for '{}'".format(key))

        [name, policy, parameters] = PolicyCollection.policy_def[key]
        return PolicyCollection.convert_to_batch(name, policy, parameters)

    @staticmethod
    def convert_to_batch(name, policy, parameters):
        """This method converts definition to a complete batch."""

        n = len(parameters)
        batch_counter = n * [0]
        batch_dims = [len(parameters[index][1]) for index in range(n)]
        batch = list()
        batch_dims[0] += 1

        # iterate over all batches
        while batch_counter[0] < batch_dims[0] - 1:

            # add the current value
            param_dict = dict()
            value_list = list()
            for k in range(n):
                current_param = parameters[k]
                element_name = current_param[0]
                element_value = current_param[1][batch_counter[k]]
                param_dict[element_name] = element_value
                value_list.append(element_value)

            single_name = name.format(*value_list)
            batch.append([single_name, policy, param_dict])

            # move to the next combination
            for k in reversed(range(n)):
                if batch_counter[k] != batch_dims[k] - 1:
                    batch_counter[k] += 1
                    break

                batch_counter[k] = 0

        return batch