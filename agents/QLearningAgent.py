# simply import the numpy package.
import tensorflow as tf

from policies.GreedyPolicy import GreedyPolicy
from spaces.DiscreteSpace import DiscreteSpace


class QLearningAgent:
    """This class represents a basic tabular q learning agent."""

    def __init__(self, session, name, state_space, action_space, policy, config, debug=False):
        """Constructs a QLearningAgent.

        Args:
            state_space: Give the discrete state space
            action_space: Give the discrete action space
            policy: Give the policies_nn the agent should use
        """
        # check if both    spaces are derived from the correct type
        assert isinstance(state_space, DiscreteSpace)
        assert isinstance(action_space, DiscreteSpace)

        # first of all save the configuration
        self.config = config

        # now the default values for some parameters can be defined.
        self.default_value('num_heads', 1)
        self.default_value('num_models', 0)
        self.default_value('optimistic', False)

        # The first two dimensions have to be defined, because it's the policies

        # set the internal debug variable
        self.deb = debug
        self.iteration = 0
        self.session = session
        self.config = config

        # Save the epsilon for the greedy policies_nn.
        self.policy = policy
        self.state_space = state_space
        self.action_space = action_space

        # define the variable scope
        with tf.variable_scope(name):

            # determine the current q values and q value for the selected head
            num_models = self.config['num_models']
            self.model_range = tf.range(0, num_models, 1)

            # Receive the initializer and use it to create a new
            # Q-Tensor
            init = self.create_initializer()
            q_tensor, q, current_heads, self.change_head = self.create_q_tensor(init)

            # Define the learning rate and discount parameters a hyper parameters
            # of the Q-Learning algorithm itself
            self.lr = tf.placeholder(tf.float64, shape=[1])
            self.discount = tf.placeholder(tf.float64, shape=[1])

            # Furthermore define the placeholders for the tuple used by one
            # learning step.
            self.current_states = tf.placeholder(tf.int32, shape=[num_models])
            self.rewards = tf.placeholder(tf.float64, shape=[num_models])
            self.actions = tf.placeholder(tf.int32, shape=[num_models])
            self.next_states = tf.placeholder(tf.int32, shape=[num_models])

            # create the operations for the density model
            cb_densities, cb_density_values, cb_step_value, cb_update \
                = self.create_density_models(self.current_states, self.actions)

            # shape the reward
            shaped_reward = tf.expand_dims(self.rewards, 1)

            # add ucb term if it is activated in the config
            if self.is_activated('ucb'):
                config['ucb-term'] = self.ucb_term(config['p'], cb_step_value, cb_density_values)

            # the same for the pseudo-count term
            if self.is_activated('pseudo-count'):
                config['pseudo-count-term'] = self.pseudo_count_term(config['beta'], cb_density_values)

            current_q_indices = tf.stack([self.model_range, self.current_states, self.actions], axis=1)
            current_q_values = tf.gather_nd(q_tensor, current_q_indices)

            # determine the q values for the next state and the selected head
            next_q_vector_indices = tf.stack([self.model_range, self.next_states], axis=1)
            next_q_vectors = tf.gather_nd(q_tensor, next_q_vector_indices)

            # get vector of best actions for each head
            next_best_q_values = tf.reduce_max(next_q_vectors, axis=1)

            # determine the TD-Error for the Q function
            td_errors = shaped_reward + self.discount * next_best_q_values - current_q_values

            # add dependencies
            with tf.control_dependencies([cb_update]):

                # define the q tensor update for the q values
                self.q_tensor_update = tf.scatter_nd_add(q_tensor, current_q_indices, self.lr * td_errors)

            # retrieve the actions here
            indices = tf.stack([self.model_range, self.current_states], axis=1)
            q_vector = tf.gather_nd(q, indices)
            self.best_actions, _ = GreedyPolicy().select_action(q_vector, None)
            self.normal_actions, _ = policy().select_action(q_vector, config)

    def ucb_term(self, p_value, cb_step_value, cb_density_value):

        # Add UCB if necessary
        p = tf.constant(p_value, dtype=tf.float64)
        return tf.sqrt(tf.div(tf.multiply(p, tf.log(cb_step_value)), (cb_density_value + 1)))

    def pseudo_count_term(self, beta_value, cb_density_value):

        beta = tf.constant(beta_value, dtype=tf.float64)
        return beta / tf.sqrt(cb_density_value + 1)

    def default_value(self, param, value):
        """This can be used to set a default value in the internal
        configuration, if no such is present their.

        Args:
            param: A string how the parameter is called
            value: The value to add to the internal config.
        """

        if param not in self.config: self.config[param] = value

    def is_activated(self, param):

        return param in self.config and self.config[param]

    def create_initializer(self):
        """This method delivers the initializer, which strongly depends
        on external configuration parameters.

        """

        # access the number of heads inside of the config
        num_heads = self.config['num_heads']
        optimistic = self.config['optimistic']
        num_models = self.config['num_models']

        # extract space sizes
        sah_list = [num_models, self.state_space.get_size(), self.action_space.get_size(), num_heads]

        # select based on the settings the correct optimization values
        if num_heads > 1 or optimistic:
            init = tf.random_normal(sah_list, dtype=tf.float64) * 20.0 + 50.0
        else:
            init = tf.zeros(sah_list, dtype=tf.float64)

        # create the initializer
        if optimistic:
            one_table = tf.ones(sah_list, dtype=tf.float64)
            init += tf.multiply(one_table, 200.0)

        return init

    def create_q_tensor(self, init):
        """This method creates the q heads and delivers a method to access one individual
        head, changeable using a sample mechanism."""

        # save the variable internally
        num_heads = self.config['num_heads']
        num_models = self.config['num_models']

        # create the q tensor
        q_tensor = tf.get_variable("q_tensor", dtype=tf.float64, initializer=init)

        # Create a sampler for the head
        random_head_num = tf.random_uniform([num_models], 0, num_heads, dtype=tf.int32)
        current_heads = tf.get_variable("current_head", initializer=random_head_num, dtype=tf.int32)
        change_head = tf.assign(current_heads, random_head_num)

        # pass back the q tensor and action to change the head
        indices = tf.stack([self.model_range, current_heads], axis=1)
        q_functions = tf.gather_nd(tf.transpose(q_tensor, [0, 3, 1, 2]), indices)

        return q_tensor, q_functions, current_heads, change_head

    def create_density_models(self, states, actions):
        """This method creates count-based density and step models for use in
        different policies of the algorithm."""

        # extract space sizes
        size_as = self.action_space.get_size()
        size_ss = self.state_space.get_size()
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

    def choose_best_action(self, cs):

        # apply defined policies_nn from the outside
        return self.session.run(self.best_action, feed_dict={self.current_state: [cs]})

    def choose_action(self, cs):

        # apply defined policies_nn from the outside
        _, res = self.session.run(self.normal_action, feed_dict={self.current_state: [cs]})
        return res

    def sample_head(self):
        """This method can be used to sample a new head for action selection. This is
        the mechanism, which effectively drives the deep exploration in the example."""

        self.session.run(self.change_head)

    def learn_tuple(self, learning_rate, discount, current_state, reward, action, next_state):
        """This method gets used to learn a new tuple and insert in the internal model.

        Args:
            learning_rate: The learning rate to use.
            discount: This determines the horizon the agent thinks about as the most important one.
            current_state: Pass in the current state of the agent.
            reward: The reward for the action in current_state
            action: The action which was taken.
            next_state: The state the agent is now in.
        """

        # update the table correctly
        self.session.run(self.q_tensor_update, feed_dict=
                                 {
                                     self.lr: learning_rate,
                                     self.discount: discount,
                                     self.current_state: [current_state],
                                     self.reward: reward,
                                     self.action: action,
                                     self.next_state: next_state
                                 })
