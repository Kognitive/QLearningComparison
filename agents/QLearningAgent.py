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
        self.name = name

        # now the default values for some parameters can be defined.
        self.default_value('num_heads', 1)
        self.default_value('num_models', 0)
        self.default_value('discount', 0.99)
        self.default_value('optimistic', False)
        self.default_value('heads_per_sample', self.config['num_heads'])

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

            # do some initialisation stuff
            if 'create_density' in self.config and self.config['create_density']:
                self.ref_density, self.cb_density = self.init_submodules()
            init = self.create_initializer()

            # obtain the q_tensor, the active q function head, the indices of the selected heads
            # as well as an operation to change the head
            model_range = tf.range(0, num_models, dtype=tf.int64)
            q_tensor, current_heads, change_head = self.create_q_tensor(init)
            ind_active_heads, sample_heads = self.get_head_indices(num_models, num_heads, heads_per_sample)

            # when shared learning is active
            if not self.is_activated('shared_learning'):
                ind_best_heads = current_heads
            else:
                ind_best_heads = tf.Variable(tf.zeros([num_models], dtype=tf.int64))

            self.q_tensor = q_tensor

            # this action can be used to sample new heads
            self.sample_heads = tf.group(change_head, sample_heads) if sample_heads is not None else change_head

            # Define the learning rate and discount parameters a hyper parameters
            # of the Q-Learning algorithm itself
            self.lr = tf.Variable(1.0, dtype=tf.float64)
            self.use_best = tf.placeholder(tf.bool, shape=None)
            discount = tf.constant(config['discount'], tf.float64)

            # Retrieve the current state from the environment and save it as an
            # expanded vector
            current_states = environment.get_current_state()
            rewards = environment.get_rewards()

            if not('ucb_infogain' in config and self.config['ucb_infogain']):

                exp_current_states = tf.expand_dims(current_states, 1)

                # Get indices for easy access of the various models
                ind_mod_head = tf.stack([model_range, current_heads], axis=1)
                ind_mod_head_cstate = tf.concat([ind_mod_head, exp_current_states], axis=1)

                # Access the q_vectors associated with the current state of each model.
                # use these to select the best and the normal action using the supplied
                # policies
                q_vector = tf.gather_nd(q_tensor, ind_mod_head_cstate)

            else:
                ind_mod_head = tf.stack([model_range, current_states], axis=1)
                ext_q_tensor = tf.gather_nd(tf.transpose(q_tensor, [0, 2, 1, 3]), ind_mod_head)
                q_mean, q_var = tf.nn.moments(ext_q_tensor, axes=[1])
                q_vector = q_mean + self.config['lambda'] * q_var

                # get the info gain bonus
                soft_ext_q_tensor = tf.nn.softmax(ext_q_tensor / self.config['info_gain_temp'])
                avg_soft_ext_q_tensor = tf.expand_dims(tf.reduce_mean(soft_ext_q_tensor, axis=1), 1)
                kl_divergence = soft_ext_q_tensor * tf.log(soft_ext_q_tensor / avg_soft_ext_q_tensor)
                self.config['info_gain_term'] = tf.reduce_mean(tf.reduce_sum(kl_divergence, axis=2), axis=1)

            self.best_actions, _ = GreedyPolicy().select_action(q_vector, None)
            self.normal_actions, _ = policy().select_action(q_vector, config)

            # Select the actions conditioned on the
            actions = tf.cond(self.use_best,
                              lambda: tf.identity(self.best_actions),
                              lambda: tf.identity(self.normal_actions))

            dep_list = list()
            dep_list.append(tf.no_op())
            if 'create_density' in self.config and self.config['create_density']:

                # Get access to the density models
                self.cb_complete_densities, cb_density_values, cb_step_value, \
                    self.ref_complete_densities, dependency_list = \
                    self.get_density_models(current_states, actions, ind_active_heads)

                # Reduce the density models according to the currently active heads
                red_indices = tf.stack([model_range, current_heads], axis=1)
                red_cb_density_values = tf.gather_nd(cb_density_values, red_indices)
                red_cb_step_value = tf.gather_nd(cb_step_value, red_indices)
                dep_list = dependency_list

            with tf.control_dependencies(dep_list):

                # get operation to perform a action on the graph
                self.apply_actions, next_states = environment.perform_actions(actions)
                if 'create_density' in self.config and self.config['create_density']:
                    rewards = self.get_shaped_rewards(rewards, red_cb_step_value, red_cb_density_values)

                self.q_tensor_update = self.get_q_tensor_update(
                    ind_active_heads, ind_best_heads, q_tensor, discount, self.lr,
                    current_states, actions, rewards, next_states, self.apply_actions)

            if self.is_activated("shared_learning"):
                ind_mod_head = tf.stack([model_range, current_states, actions], axis=1)
                all_heads = tf.gather_nd(tf.transpose(q_tensor, [0, 2, 3, 1]), ind_mod_head)
                best_heads = tf.argmax(all_heads, axis=1)
                self.get_best_heads = tf.assign(ind_best_heads, best_heads)

    def get_density_models(self, current_states, actions, ind_active_heads):

        # access some vars from the config
        num_models = self.config['num_models']
        num_heads = self.config['num_heads']
        heads_per_sample = self.config['heads_per_sample']

        model_range = tf.range(0, num_models, dtype=tf.int64)
        ind_models = self.duplicate_each_element(model_range, heads_per_sample)

        # This reference is necessary to evaluate the graph for the density model
        cb_var = tf.get_variable("real_var", [], initializer=tf.zeros_initializer, dtype=tf.float64)

        if ind_active_heads is not None:
            mask_indices = tf.stack([ind_models, ind_active_heads], axis=1)
            current_head_mask_tensor = tf.scatter_nd(mask_indices,
                                                     tf.ones([num_models * heads_per_sample], dtype=tf.int64),
                                                     shape=[num_models, num_heads])
        else:
            current_head_mask_tensor = tf.ones([num_models, num_heads], dtype=tf.int64)

        # save the assign operations in a dependency list
        dependency_list = list()

        with tf.variable_scope("opt_cb_dens"):
            cb_complete_densities, cb_density_values, cb_step_value = \
                self.ref_density.get_graph(current_states, actions, current_head_mask_tensor)
            update_cb_density_model = tf.assign(cb_var, tf.reduce_sum(cb_density_values))
            dependency_list.append(update_cb_density_model)
            ref_complete_densities = cb_complete_densities

        # check whether a approximate density model should be activated
        if self.is_activated('pseudo_count') and self.config['pseudo_count_type'] != 'count_based':

            approx_var = tf.get_variable("approx_var", [], initializer=tf.zeros_initializer, dtype=tf.float64)
            ref_complete_densities = cb_complete_densities

            with tf.variable_scope("opt_approx_dens"):
                approx_complete_densities, approx_density_values, approx_step_value = \
                    self.cb_density.get_graph(current_states, actions, current_head_mask_tensor)
                update_approx_density_model = tf.assign(approx_var, tf.reduce_sum(approx_density_values))
                dependency_list.append(update_approx_density_model)

                # use the approximate as the reference model
                cb_complete_densities = approx_complete_densities
                cb_density_values = approx_density_values
                cb_step_value = approx_step_value

        return cb_complete_densities, cb_density_values, cb_step_value, ref_complete_densities, dependency_list

    def get_shaped_rewards(self, rewards, current_steps, current_state_action_counts):

        # shape the reward
        shaped_rewards = rewards

        # add ucb term if it is activated in the config
        if self.is_activated('ucb'):
            self.config['ucb-term'] = self.ucb_term(self.config['p'], current_steps, current_state_action_counts)
            shaped_rewards += self.config['ucb-term']

        # the same for the pseudo-count term
        if self.is_activated('pseudo_count'):
            self.config['pseudo-count-term'] = self.pseudo_count_term(self.config['beta'], current_state_action_counts)
            shaped_rewards += self.config['pseudo-count-term']

        if 'info_gain_term' in self.config:
            shaped_rewards += self.config['info_gain_term']

        return shaped_rewards

    def get_q_tensor_update(self, ind_heads, ind_best_heads, q_tensor, discount, lr, current_states, actions, rewards, next_states, apply_actions):

        num_models = self.config['num_models']
        num_heads = self.config['num_heads']
        heads_per_sample = self.config['heads_per_sample']
        model_range = tf.range(0, num_models, dtype=tf.int64)

        # we have to modify the states and actions a little bit
        ind_models = self.duplicate_each_element(model_range, heads_per_sample)
        ind_states = self.duplicate_each_element(current_states, heads_per_sample)
        ind_best_heads_duplic = tf.cast(self.duplicate_each_element(ind_best_heads, heads_per_sample), tf.int64)
        ind_actions = self.duplicate_each_element(actions, heads_per_sample)
        ind_next_states = self.duplicate_each_element(next_states, heads_per_sample)

        if ind_heads == None:
            ind_heads = tf.tile(tf.range(0, num_heads, dtype=tf.int64), [num_models])

        # obtain current q values
        ind_current_q_values = tf.stack([ind_models, ind_heads, ind_states, ind_actions], axis=1)
        current_q_values = tf.gather_nd(q_tensor, ind_current_q_values)

        # retrieve the best model
        ind_best_q_values = tf.stack([ind_models, ind_best_heads_duplic, ind_states], axis=1)
        best_q_values = tf.gather_nd(q_tensor, ind_best_q_values)
        actions = tf.argmax(best_q_values, axis=1)

        # obtain the best q function available for the next state
        ind_next_q_vectors = tf.stack([ind_models, ind_heads, ind_next_states, tf.squeeze(actions)], axis=1)
        next_q_values = tf.gather_nd(q_tensor, ind_next_q_vectors)

        # duplicate the rewards as well
        mod_shaped_rewards = self.duplicate_each_element(rewards, heads_per_sample)
        td_errors = mod_shaped_rewards + discount * next_q_values - current_q_values

        # add dependencies
        with tf.control_dependencies([apply_actions]):

            # define the q tensor update for the q values
            return tf.scatter_nd_add(q_tensor, ind_current_q_values, lr * td_errors)

    def duplicate_each_element(self, vector: tf.Tensor, repeat: int) -> tf.Tensor:
        """This method takes a vector and duplicates each element the number of times supplied."""

        height = tf.shape(vector)[0]
        exp_vector = tf.expand_dims(vector, 1)
        tiled_states = tf.tile(exp_vector, [1, repeat])
        mod_vector = tf.reshape(tiled_states, [repeat * height])
        return mod_vector

    def get_head_indices(self, num_models, num_heads, heads_per_sample):

        if heads_per_sample == num_heads: return None, None

        head_range = tf.range(0, num_heads, dtype=tf.int64)
        randomized_head_range_list = [tf.random_shuffle(head_range)[:heads_per_sample] for i in range(num_models)]
        head_indices = tf.concat(randomized_head_range_list, axis=0)
        current_head_indices = tf.Variable(head_indices, dtype=tf.int64)
        sample_head_indices = tf.assign(current_head_indices, head_indices)

        return current_head_indices, sample_head_indices

    def init_submodules(self):

        density_config = {'num_models': self.config['num_models'], 'num_heads': self.config['num_heads'],
                          'action_space': self.action_space,  'state_space': self.state_space,
                          'num_hidden': 40, 'heads_per_sample': self.config['heads_per_sample']}

        ref_density = CountBasedModel(density_config)
        cb_density = None

        if self.is_activated('pseudo_count') and self.config['pseudo_count_type'] != 'count_based':

            # create the operations for the density model
            density_config['learning_rate'] = self.config['learning_rate']
            density_model = NADEModel(density_config)
            cb_density = CountBasedAdapter(self.config, density_model)

        return ref_density, cb_density

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
        num_models = self.config['num_models']
        num_heads = self.config['num_heads']

        # extract space sizes
        sah_list = [num_models, num_heads, self.state_space.get_size(), self.action_space.get_size()]

        # select based on the settings the correct optimization values
        if self.is_activated('init_zero'):
            init = tf.zeros(sah_list, dtype=tf.float64)

        elif not self.is_activated('optimistic'):
            offset = 30
            mu = (2 * self.config['min_q'] - offset) / 2
            sigma = offset / 2

            if self.is_activated('init_gaussian'):
                init = tf.random_normal(sah_list, mu, sigma, dtype=tf.float64)

            else:
                init = tf.random_uniform(sah_list, self.config['min_q'] - offset, self.config['min_q'], dtype=tf.float64)

        elif self.is_activated('optimistic'):
            sigma = 10
            mu = self.config['max_q'] + 2 * sigma
            init = tf.random_uniform(sah_list, self.config['max_q'] + sigma, self.config['max_q'] + 2 * sigma, dtype=tf.float64)

        return init

    def create_q_tensor(self, init):
        """This method creates the q heads and delivers a method to access one individual
        head, changeable using a sample mechanism."""

        # save the variable internally
        num_models = self.config['num_models']
        num_heads = self.config['num_heads']

        # create the q tensor
        q_tensor = tf.get_variable("q_tensor", dtype=tf.float64, initializer=init)

        # Create a sampler for the head
        random_head_num = tf.random_uniform([num_models], 0, num_heads, dtype=tf.int64)
        current_heads = tf.get_variable("current_head", initializer=random_head_num, dtype=tf.int64)
        change_head = tf.assign(current_heads, random_head_num)

        return q_tensor, current_heads, change_head

    def sample_head(self):
        """This method can be used to sample a new head for action selection. This is
        the mechanism, which effectively drives the deep exploration in the example."""

        self.session.run(self.sample_heads)