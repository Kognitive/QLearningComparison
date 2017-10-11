# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from spaces.DiscreteSpace import DiscreteSpace


class DeterministicMDP:
    """This represents a deterministic MDP"""

    def __init__(self, name, state_tensor_shape, action_space, state_space, transition_function, reward_function, initial_state):
        """Construct a new general deterministic MDP.

        Args:
            state_tensor_shape: A list [d_0, ..., d_{N-1}] containing the shape of the state tensor
            action_space: The action space to use, has to be derived from DiscreteSpace as well.
            state_space: The state space to use, it has to be derived from DiscreteSpace as well.
            transition_function: The function, which maps state, actions to states matrix.
            reward_function: The reward matrix, stating the reward for each state, action pair.
            initial_state: The initial state
        """

        # check if the parameters are indeed the correct instances
        assert isinstance(action_space, DiscreteSpace)
        assert isinstance(state_space, DiscreteSpace)
        assert isinstance(initial_state, int) and 0 <= initial_state < state_space.get_size()

        # save the number of states
        self.state_tensor_shape = state_tensor_shape
        self.dim = len(state_tensor_shape)
        self.initial_state = initial_state
        self.action_space = action_space
        self.state_space = state_space
        self.name = name
        self.q_function = None
        self.optimal_reward = None

        with tf.variable_scope("env_{}".format(name)):
            if isinstance(reward_function, np.ndarray):

                # normalize the reward function
                mi = np.min(reward_function)
                ma = np.max(reward_function)
                reward_function = (reward_function - mi) / (ma - mi)

                # Do some assertions on the passed reward and transition functions.
                # They need to have the height of the state space and the width of
                state_action_shape = (state_space.get_size(), action_space.get_size())
                assert np.shape(transition_function) == state_action_shape
                assert np.shape(reward_function) == state_action_shape

                # check if transition function is valid
                for i in range(np.size(transition_function, 0)):
                    for j in range(np.size(transition_function, 1)):
                        assert 0 <= transition_function[i, j] < state_space.get_size()

                # save passed parameters
                self.transition = tf.constant(transition_function, dtype=tf.int64)
                self.rewards = tf.constant(reward_function, dtype=tf.float64)
                self.reward_function = reward_function
                self.transition_function = transition_function

            else:
                self.transition = transition_function
                self.rewards = reward_function

            # Create the current state vector as well as the operation to reset it to the initial state
            init = tf.constant(self.initial_state, shape=state_tensor_shape, dtype=tf.int64)
            self.current_states = tf.get_variable("current_state", dtype=tf.int64, initializer=init)
            self.cum_rewards = tf.get_variable("cum_rewards", state_tensor_shape, dtype=tf.float64, initializer=tf.zeros_initializer)
            self.eps_rewards = tf.get_variable("eps_rewards", state_tensor_shape, dtype=tf.float64, initializer=tf.zeros_initializer)

            reset_state = tf.assign(self.current_states, init)
            zero_const = tf.constant(0.0, shape=state_tensor_shape, dtype=tf.float64)
            reset_cum_rewards = tf.assign(self.cum_rewards, zero_const)
            reset_eps_rewards = tf.assign(self.eps_rewards, zero_const)

            self.reset_op = tf.group(reset_state, reset_cum_rewards, reset_eps_rewards)

    def get_current_state(self):
        return self.current_states

    def get_rewards(self):
        return self.eps_rewards

    def perform_actions(self, actions):

        # access the state action values inside of the transition function
        selection = tf.stack([self.current_states, actions], axis=self.dim)
        next_state = tf.gather_nd(self.transition, selection)
        rewards = tf.gather_nd(self.rewards, selection)
        ass_curr_state = tf.assign(self.current_states, next_state)
        ass_coll_rewards = tf.assign_add(self.cum_rewards, rewards)
        ass_eps_rewards = tf.assign(self.eps_rewards, rewards)

        # save the reward and update state
        return tf.group(ass_curr_state, ass_coll_rewards, ass_eps_rewards), next_state

    def clone(self, new_name):
        return DeterministicMDP(new_name, self.state_tensor_shape,
                                self.action_space, self.state_space,
                                self.transition_function, self.reward_function,
                                self.initial_state)

    def get_optimal(self, steps, discount):
        """This gets the optimal reward using value iteration."""

        if self.q_function is None:

            state_size = self.state_space.get_size()
            action_size = self.action_space.get_size()

            # init q function
            q_shape = (state_size, action_size)
            q_function = -np.ones(q_shape)
            next_q_function = np.zeros(q_shape)

            # repeat until converged
            while np.max(np.abs(q_function - next_q_function)) >= 0.001:

                # create next bootstrapped q function
                q_function = next_q_function
                bootstrapped_q_function = np.empty(q_shape)

                # iterate over all fields
                for s in range(state_size):
                    for a in range(action_size):
                        next_state = self.transition_function[s, a]
                        bootstrapped_q_function[s, a] = np.max(q_function[next_state, :])

                # update the q function correctly
                next_q_function = self.reward_function + discount * bootstrapped_q_function

            # create new environment and simulate
            optimal_policy = np.argmax(q_function, axis=1)
            reward = 0
            current_state = self.initial_state

            # run for specified number of steps
            for k in range(steps):
                reward += self.reward_function[current_state, optimal_policy[current_state]]
                current_state = self.transition_function[current_state, optimal_policy[current_state]]

            self.optimal_reward = reward
            self.q_function = q_function

            # create new environment and simulate
            minimal_policy = np.argmin(q_function, axis=1)
            reward = 0
            current_state = self.initial_state

            # run for specified number of steps
            for k in range(steps):
                reward += self.reward_function[current_state, minimal_policy[current_state]]
                current_state = self.transition_function[current_state, minimal_policy[current_state]]

            self.minimal_reward = reward

        return self.optimal_reward, self.minimal_reward, np.min(self.q_function), np.max(self.q_function), q_function
