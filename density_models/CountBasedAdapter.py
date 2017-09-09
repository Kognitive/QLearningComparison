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
        self.density_model = density_model

    def get_graph(self, states, actions):

        # use the internal density model
        density_value, minimizer = self.density_model.get_graph(states, actions)

        # create normal count based update
        cb_step_value = tf.get_variable("cb_step", dtype=tf.int64, shape=[], initializer=tf.zeros_initializer)
        cb_step_update = tf.count_up_to(cb_step_value, 2 ** 63 - 1)

        # Place for the saved evaluation
        saved_evaluation = tf.get_variable("saved_evaluation", [self.num_models], trainable=False)
        save_evaluation = tf.assign(saved_evaluation, density_value)

        # retrieve the minimizer
        with tf.control_dependencies([save_evaluation]):
            cb_update = tf.group([cb_step_update, minimizer])

        # execute the minimizer in prior
        with tf.control_dependencies([cb_update]):

            # switch between two cases the first is the prediction gain
            if self.config['pseudo_count_type'] == 'prediction_gain':
                c = tf.constant(self.config['c'], dtype=tf.float32)
                prediction_gain = tf.log(density_value) - tf.log(saved_evaluation)
                cb_values = tf.inv(tf.exp(c * tf.pow(cb_step_value, -0.5) * prediction_gain)- 1)

            elif self.config['pseudo_count_type'] == 'pseudo_count':
                cb_values = saved_evaluation * (1 - density_value) / (density_value - saved_evaluation)

            else:
                raise RuntimeError("You have to use prediction_gain or pseudo_counts")

        return cb_values, cb_step_value