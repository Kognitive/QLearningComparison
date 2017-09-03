from spaces.DiscreteSpace import DiscreteSpace


class DiscreteEnvironment:

    def __init__(self, num_states, action_space, state_space, initial_state):
        """Initialize a new environment. Therefore one has to specify how many states
        should be stored inside simultaneously.

        Args:
            num_states: The number of individual states."""

        # check if the parameters are indeed the correct instances
        assert isinstance(action_space, DiscreteSpace)
        assert isinstance(state_space, DiscreteSpace)
        assert isinstance(num_states, int)
        assert isinstance(initial_state, int) and 0 <= initial_state < state_space.get_size()

        # save the number of states
        self.num_states = num_states
        self.initial_state = initial_state
        self.action_space = action_space
        self.state_space = state_space

        # create the current state vector
        self.current_state_vector = self.num_states * [initial_state]

    def get_action_space(self):
        """This method delivers the action space of this environment"""

        return self.action_space

    def get_state_space(self):
        """This method delivers the state space of this environment."""

        return self.state_space

    def get_states(self):

        """This method delivers the current state of this environment."""
        return self.current_state_vector

    def update_states(self, new_states):
        """This method updates the internal state vector."""

        self.current_state_vector = new_states

    def reset(self):
        """This method sets the current state vector internally. It
        basically inserts the same value for each individual state."""

        self.current_state_vector = self.num_states * [self.initial_state]

    def perform_actions(self, actions):
        """This method should perform an action on the environment.
        The expected behaviour is that the action gets executed and
        as a return the callee receives the reward.

        Args:
            actions: A list of actions to execute

        Returns:
            A list of rewards received for performing action[i] on state[i]
        """
        raise NotImplementedError("You have to implement a action method.")

    def get_states(self):
        """This method delivers the state vector of the environment.

        Returns:
            A list of states for each individual environment
        """
        return self.current_state_vector

    def get_optimal(self, steps, discount):
        """This method obtains the optimal reward for the infinite horizon case.
        It therefore needs the number of steps and the discount factor.

        Args:
            steps: The number of steps to run the policy.
            discount: The discount factor.
        """

        raise NotImplementedError("Your environment has to pass back an optimal reward value.")

    def actions(self, actions):
        """This method should perform an action on the environment.
        The expected behaviour is that the action gets executed and
        as a return the callee receives the reward.

        Args:
            actions: A list of actions to execute

        Returns:
            A list of rewards received for these actions
        """
        return self.perform_actions(actions)