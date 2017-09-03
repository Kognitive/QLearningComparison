import tensorflow as tf

from policies.Policy import Policy


class GreedyPolicy(Policy):
    """This policies_nn selects the best action the model predicts."""

    def choose_action(self, q, config):
        """Creates a graph, where the best action is selected.

        Args:
            q: The q function to use for evaluating.
        """

        # get the number of states
        actions = tf.cast(tf.argmax(q, axis=1), tf.int32)
        max_q = tf.reduce_max(q, axis=1)

        return actions, max_q
