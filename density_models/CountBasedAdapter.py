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


class CountBasedAdapter:
    """This class takes a density model and is able to produce"""

    def __init__(self, config, density_model):
        """Creates a new CountBasedAdapter.

        Args:
            config:
                count_type: prediction_gain || pseudo_count
                density_model: The density model to use for this CountBasedAdapter"""

        self.config = config
        self.num_models = config['num_models']
        self.num_heads = config['num_heads']
        self.density_model = density_model

    def get_graph(self, states: tf.Tensor, actions: tf.Tensor, head_mask_tensor: tf.Tensor):

        # create normal count based update
        cb_step_value = tf.get_variable("cb_step", dtype=tf.int64, shape=[self.num_models, self.num_heads], initializer=tf.ones_initializer)
        cb_step_update = tf.assign_add(cb_step_value, head_mask_tensor)

        # use the internal density model
        old_density_value, minimizer = self.density_model.get_graph(states, actions, head_mask_tensor, True)
        cb_update = tf.group(cb_step_update, minimizer)

        with tf.control_dependencies([cb_update]):
            density_value = self.density_model.get_graph(states, actions, head_mask_tensor, False)
            all_densities = self.density_model.get_all_densities()

            # switch between two cases the first is the prediction gain
            if self.config['pseudo_count_type'] == 'prediction_gain':
                c = tf.constant(self.config['c'], dtype=tf.float64)
                prediction_gain = tf.log(density_value) - tf.log(old_density_value)
                prediction_gain = prediction_gain
                cb_values = 1 / (tf.exp(c * tf.pow(tf.sqrt(tf.cast(cb_step_value, tf.float64)), -1) * prediction_gain) - 1)

            elif self.config['pseudo_count_type'] == 'pseudo_count':
                cb_values = old_density_value * (1 - density_value) / (density_value - old_density_value)

        return all_densities, cb_values, cb_step_value