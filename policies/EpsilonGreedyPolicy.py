import tensorflow as tf
from policies.Policy import Policy

from policies.GreedyPolicy import GreedyPolicy


class EpsilonGreedyPolicy(Policy):
    """This represents a simple epsilon greedy policies_nn. What it basically does,
    is, that it selects the best action in 1 - epsilon of the cases and in
    epsilon cases it wil basically select a random one.
    """

    def choose_action(self, q, config):
        """Create the tree for epsilon greedy policies_nn selection.

        Args:
            q: The q function to use for evaluating.
            config: The configuration for this epsilon greedy policies_nn

        Returns: The tensorflow graph
        """

        # get the number of states
        best_actions, _ = GreedyPolicy().select_action(q, config)

        # sample a random decision vector
        num_models = tf.cast(tf.shape(q)[0], tf.int64)

        # sample some random values
        sample_size = [num_models]
        random_actions = tf.random_uniform(sample_size, maxval=config['action_space'].get_size(), dtype=tf.int64)
        random_decision_vector = tf.less(tf.random_uniform(sample_size), config['epsilon'])

        # let the coin decide about the final actions
        final_actions = tf.where(random_decision_vector, random_actions, best_actions)

        # pass back the actions and corresponding q values
        model_range = tf.range(0, num_models, 1, dtype=tf.int64)
        indices = tf.stack([model_range, final_actions], axis=1)
        q_values = tf.gather_nd(q, indices)
        return final_actions, q_values