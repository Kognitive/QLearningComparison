# simply import the numpy package.
import tensorflow as tf
import collections

from policies.GreedyPolicy import GreedyPolicy
from spaces.DiscreteSpace import DiscreteSpace
from environments.DeterministicMDP import DeterministicMDP
from density_models.CountBasedModel import CountBasedModel
from density_models.CountBasedAdapter import CountBasedAdapter
from density_models.NADEModel import NADEModel

class QLearningAgent:
    """This class represents a basic tabular q learning agent."""

    def __init__(self, session, name, environment, policy, config, debug=False):
        """Constructs a QLearningAgent.

        Args:
            session: The session the agent should effectively use.
            name: The unique name for this QLearningAgent
            policy:
        """

        # check if both    spaces are derived from the correct type
        assert isinstance(environment, DeterministicMDP)

        # first of all save the configuration
        self.config = config

        # now the default values for some parameters can be defined.
        self.default_value('num_heads', 1)
        self.default_value('num_models', 0)
        self.default_value('optimistic', False)
        self.default_value('heads_per_sample', self.config['num_heads'])

        # The first two dimensions have to be defined, because it's the policies

        # set the internal debug variable
        self.deb = debug
        self.iteration = 0
        self.session = session
        self.config = config

        # check whether the settings are indeed correct
        heads_per_sample = self.config['heads_per_sample']
        num_heads = self.config['num_heads']
        num_models = self.config['num_models']
        assert 1 <= heads_per_sample <= num_heads

        # Save the epsilon for the greedy policies_nn.
        self.policy = policy
        self.state_space = environment.state_space
        self.action_space = environment.action_space

        # define the variable scope
        with tf.variable_scope(name):

            self.init_submodules()

            # determine the current q values and q value for the selected head
            num_models = self.config['num_models']
            self.model_range = tf.range(0, num_models, 1, dtype=tf.int64)

            # Receive the initializer and use it to create a new
            # Q-Tensor
            init = self.create_initializer()
            q_tensor, q, current_heads, self.change_head = self.create_q_tensor(init)

            # Define the learning rate and discount parameters a hyper parameters
            # of the Q-Learning algorithm itself
            self.lr = tf.constant(1, tf.float64)
            self.discount = tf.constant(0.99, tf.float64)
            self.use_best = tf.placeholder(tf.bool, shape=None)

            # Furthermore define the placeholders for the tuple used by one
            # learning step.
            self.current_states = environment.get_current_state()

            # retrieve the actions here
            indices = tf.stack([self.model_range, self.current_states], axis=1)
            q_vector = tf.gather_nd(q, indices)
            self.best_actions, _ = GreedyPolicy().select_action(q_vector, None)
            self.normal_actions, _ = policy().select_action(q_vector, config)

            # select the actions it should use
            self.actions = tf.cond(self.use_best, lambda: tf.identity(self.best_actions), lambda: tf.identity(self.normal_actions))
            self.rewards = environment.get_rewards()

            # create a dependency list
            dependency_list = list()

            # create a sampling routine for the head mask tensor
            if heads_per_sample != num_heads:

                model_vector = tf.where(tf.range(0, num_heads) < heads_per_sample,
                                        tf.ones(num_heads, dtype=tf.int64),
                                        tf.zeros(num_heads, dtype=tf.int64))

                model_vector_list = [tf.random_shuffle(model_vector) for _ in range(num_models)]
                head_mask_tensor = tf.stack(model_vector_list, axis=0)
                current_head_mask_tensor = tf.get_variable("current_head_mask_tensor", [num_models, num_heads], dtype=tf.int64)
                sample_new_head_mask = tf.assign(current_head_mask_tensor, head_mask_tensor)
                dependency_list.append(sample_new_head_mask)

            else:
                current_head_mask_tensor = tf.ones([num_models, num_heads], dtype=tf.int64)

            # create reference if necessary
            ref_placeholder = tf.get_variable("ref_dens", [], trainable=False, initializer=tf.zeros_initializer, dtype=tf.float64)

            if self.is_activated('pseudo_count') and self.config['pseudo_count_type'] != 'count_based':

                # create the reference density model as well as the density for the learned model
                with tf.variable_scope("opt_ref_dens"):
                    self.all_densities, ref_density_values, ref_cb_value = self.ref_density.get_graph(self.current_states, self.actions, current_head_mask_tensor)
                    update_density_model = tf.assign(ref_placeholder, tf.reduce_sum(ref_density_values))
                    dependency_list.append(update_density_model)

                with tf.variable_scope("opt_approx_dens"):
                    self.all_densities_model, cb_density_values, cb_step_value = self.cb_density.get_graph(self.current_states, self.actions, current_head_mask_tensor)

                cb_placeholder = tf.get_variable("cb_dens", [], trainable=False, initializer=tf.zeros_initializer, dtype=tf.float64)
                update_density_model = tf.assign(cb_placeholder, tf.reduce_sum(cb_density_values))
                dependency_list.append(update_density_model)

            else:
                self.all_densities_model, cb_density_values, cb_step_value = self.ref_density.get_graph(self.current_states, self.actions, current_head_mask_tensor)
                self.all_densities = self.all_densities_model
                update_density_model = tf.assign(ref_placeholder, tf.reduce_sum(cb_density_values))
                dependency_list.append(update_density_model)

            # Reduce the density models according to the currently active heads
            red_indices = tf.stack([self.model_range, current_heads], axis=1)
            red_cb_density_values = tf.gather_nd(cb_density_values, red_indices)
            red_cb_step_value = tf.gather_nd(cb_step_value, red_indices)

            with tf.control_dependencies(dependency_list):
                perform_operation, self.next_states = environment.perform_actions(self.actions)

            # shape the reward
            shaped_reward = tf.expand_dims(self.rewards, 1)

            # add ucb term if it is activated in the config
            if self.is_activated('ucb'):
                config['ucb-term'] = self.ucb_term(config['p'], red_cb_step_value, red_cb_density_values)
                shaped_reward += tf.expand_dims(config['ucb-term'], axis=1)

            # the same for the pseudo-count term
            if self.is_activated('pseudo_count'):
                config['pseudo-count-term'] = self.pseudo_count_term(config['beta'], red_cb_density_values)
                shaped_reward += tf.expand_dims(config['pseudo-count-term'], axis=1)

            # create the matching indices for the head
            head_indices = tf.cast(tf.where(tf.equal(current_head_mask_tensor, tf.constant(1, dtype=tf.int64))), tf.int64)

            # we have to modify the states and actions a little bit
            mod_size = [num_models * heads_per_sample]
            tiled_states = tf.tile(tf.expand_dims(self.current_states, 1), [1, heads_per_sample])
            tiled_actions = tf.tile(tf.expand_dims(self.actions, 1), [1, heads_per_sample])
            mod_states = tf.expand_dims(tf.reshape(tiled_states, mod_size), 1)
            mod_actions = tf.expand_dims(tf.reshape(tiled_actions, mod_size), 1)

            current_q_indices = tf.concat([head_indices, mod_states, mod_actions], axis=1)
            current_q_values = tf.gather_nd(q_tensor, current_q_indices)

            # do the same modulating for the next states as well
            tiled_next_states = tf.tile(tf.expand_dims(self.next_states, 1), [1, heads_per_sample])
            mod_next_states = tf.expand_dims(tf.reshape(tiled_next_states, mod_size), 1)

            # determine the q values for the next state and the selected head
            next_q_vector_indices = tf.concat([head_indices, mod_next_states], axis=1)
            next_q_vectors = tf.gather_nd(q_tensor, next_q_vector_indices)

            # get vector of best actions for each head
            next_best_q_values = tf.reduce_max(next_q_vectors, axis=1)

            # determine the TD-Error for the Q function
            tiled_shaped_reward = tf.tile(shaped_reward, [1, heads_per_sample])
            mod_shaped_reward = tf.reshape(tiled_shaped_reward, mod_size)

            td_errors = mod_shaped_reward + self.discount * next_best_q_values - current_q_values
            self.perform_operation = perform_operation

            # add dependencies
            with tf.control_dependencies([perform_operation]):

                # define the q tensor update for the q values
                self.q_tensor_update = tf.scatter_nd_add(q_tensor, current_q_indices, self.lr * td_errors), [tf.constant(0, dtype=tf.int64)]

    def init_submodules(self):

        density_config = {'num_models': self.config['num_models'], 'num_heads': self.config['num_heads'],
                          'action_space': self.action_space,  'state_space': self.state_space,
                          'num_hidden': 7, 'heads_per_sample': self.config['heads_per_sample']}

        self.ref_density = CountBasedModel(density_config)

        if self.is_activated('pseudo_count') and self.config['pseudo_count_type'] != 'count_based':

            # create the operations for the density model
            self.density_model = NADEModel(density_config)
            self.cb_density = CountBasedAdapter(self.config, self.density_model)

    def ucb_term(self, p_value, cb_step_value, cb_density_value):

        # Add UCB if necessary
        p = tf.constant(p_value, dtype=tf.float64)
        cb_step = tf.cast(cb_step_value, dtype=tf.float64)
        cb_density = tf.cast(cb_density_value, dtype=tf.float64)
        return tf.sqrt(tf.div(tf.multiply(p, tf.log(cb_step)), (cb_density + 1)))

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
        sah_list = [num_models, num_heads, self.state_space.get_size(), self.action_space.get_size()]

        # select based on the settings the correct optimization values
        if self.config['num_heads'] > 1 or self.is_activated('optimistic'):
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
        random_head_num = tf.random_uniform([num_models], 0, num_heads, dtype=tf.int64)
        current_heads = tf.get_variable("current_head", initializer=random_head_num, dtype=tf.int64)
        change_head = tf.assign(current_heads, random_head_num)

        # pass back the q tensor and action to change the head
        indices = tf.stack([self.model_range, current_heads], axis=1)
        q_functions = tf.gather_nd(q_tensor, indices)

        return q_tensor, q_functions, current_heads, change_head

    def choose_best_action(self, cs):

        # apply defined policies_nn from the outside
        return self.session.run(self.best_actions, feed_dict={self.current_states: [cs]})

    def choose_action(self, cs):

        # apply defined policies_nn from the outside
        _, res = self.session.run(self.normal_actions, feed_dict={self.current_states: [cs]})
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
                                     self.current_states: [current_state],
                                     self.rewards: reward,
                                     self.actions: action,
                                     self.next_states: next_state
                                 })
