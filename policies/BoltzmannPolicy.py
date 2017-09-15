import tensorflow as tf

from policies.Policy import Policy


class BoltzmannPolicy(Policy):
    """Represents a simple boltzmann policies_nn."""

    def choose_action(self, q, config):
        """This method of a policies_nn basically gets a Q function
        and has to return the action to take now. Here you can
        specify behaviour like taking the best action, or sometimes
        different actions to them.

        Args:
            q: The q function to use for evaluating.

        Returns: The index of the action that should be taken
        """

        # get the number of states
        soft_max = tf.nn.softmax(q / config['temperature'])

        # create the categorical
        dist = tf.distributions.Categorical(probs=soft_max)
        actions = tf.cast(dist.sample(), tf.int64)
        actions = actions

        # pass back the actions and corresponding q values
        model_range = tf.range(0, config['num_models'], 1, dtype=tf.int64)
        indices = tf.stack([model_range, actions], axis=1)
        q_values = tf.gather_nd(q, indices)
        return actions, q_values
