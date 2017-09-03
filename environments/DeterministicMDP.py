# import the necessary packages
import numpy as np

from environments.DiscreteEnvironment import DiscreteEnvironment
from spaces.DiscreteSpace import DiscreteSpace


class DeterministicMDP(DiscreteEnvironment):
    """This represents a deterministic MDP"""

    def __init__(self, N, action_space, state_space, transition_function, reward_function, initial_state):
        """Construct a new general deterministic MDP.

        Args:
            N: The number of individual states
            action_space: The action space to use, has to be derived from DiscreteSpace as well.
            state_space: The state space to use, it has to be derived from DiscreteSpace as well.
            transition_function: The function, which maps state, actions to states matrix.
            reward_function: The reward matrix, stating the reward for each state, action pair.
            initial_state: The initial state
        """

        # Call the super class
        super().__init__(N, action_space, state_space, initial_state)

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
        self.transition = transition_function
        self.rewards = reward_function

    def perform_actions(self, actions):
        """This method performs the action according to the internal table."""

        # retrieve all rewards and the next states
        current_states = self.get_states()
        current_rewards = self.rewards[current_states, actions]
        next_states = self.transition[current_states, actions]

        # update the internal state accordingly
        self.update_states(next_states)

        # pass back the rewards
        return current_rewards

    def single_clone(self):
        raise NotImplementedError("You have to supply a single clone operation.")

    def get_optimal(self, steps, discount):
        """This gets the optimal reward using value iteration."""

        state_size = self.get_state_space().get_size()
        action_size = self.get_action_space().get_size()

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
                    next_state = self.transition[s, a]
                    bootstrapped_q_function[s, a] = np.max(q_function[next_state, :])

            # update the q function correctly
            next_q_function = self.rewards + discount * bootstrapped_q_function

        # create new environment and simulate
        optimal_policy = np.argmax(q_function, axis=1)
        environment = self.single_clone()
        reward = 0

        # run for specified number of steps
        for k in range(steps):
            reward += environment.actions([optimal_policy[environment.get_states()[0]]])[0]

        return reward
