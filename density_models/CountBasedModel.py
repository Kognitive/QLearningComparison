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

import tensorflow as tf


class CountBasedModel:
    """This represents a count based model. That means the class maintains
    a table of counted density values. This is obviously not scalable to
    arbitrary scaled tasks.
    """

    def __init__(self, config):
        """Creates a new CountBasedModel using the supplied configuration

        Args:
            config:
                num_models: The number of independent density models to create.
                action_space: The action space to use for the model.
                state_space: The state space to use for the model.
            """

        self.config = config
        self.model_range = tf.range(0, self.config['num_models'], 1)

    def get_graph(self, states, actions):
        """This method creates count-based density and step models for use in
        different policies of the algorithm."""

        # extract space sizes
        size_as = self.config['action_space'].get_size()
        size_ss = self.config['state_space'].get_size()
        num_models = self.config['num_models']

        # count based density model and the current step are simple variables
        # with the appropriate sizes
        cb_density = tf.get_variable("cb_density", dtype=tf.int64, shape=[num_models, size_ss, size_as], initializer=tf.zeros_initializer)
        cb_step_value = tf.get_variable("cb_step", dtype=tf.int64, shape=[], initializer=tf.zeros_initializer)

        # create a operation which adds on the cb_step variable
        cb_step_update = tf.count_up_to(cb_step_value, 2 ** 63 - 1)

        # for each dimension in append create appropriate density values
        cb_indices = tf.stack([self.model_range, states, actions], axis=1)
        cb_density_values = tf.gather_nd(cb_density, cb_indices)
        cb_density_update = tf.scatter_nd_add(cb_density, indices=cb_indices, updates=tf.constant(1, shape=[num_models], dtype=tf.int64))

        # create a combined update action
        cb_update = tf.group(cb_density_update, cb_step_update)

        # pass back teh actions to the callee
        return cb_density, cb_density_values, cb_step_value, cb_update