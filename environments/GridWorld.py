import numpy as np

from environments.DeterministicMDP import DeterministicMDP
from spaces.DiscreteSpace import DiscreteSpace


class GridWorld(DeterministicMDP):

    def __init__(self, name, num_states, N):

        # create the state and action space
        self.inner_size = N
        state_space = DiscreteSpace(N ** 2)
        action_space = DiscreteSpace(4)

        # get size of state and action space
        size_space = state_space.get_size()
        size_action = action_space.get_size()

        # one maps to 2
        starting_state = 0

        # specify the transition function
        transition_func = np.zeros((size_space, size_action), dtype=np.int32)
        reward_function = np.zeros((size_space, size_action), dtype=np.float64)

        # iterate over and fill with the transitions
        for i in range(size_space):
            transition_func[i, 0] = i if i % N == 0 else i - 1
            transition_func[i, 1] = i if i / N < 1 else i - N
            transition_func[i, 2] = i if i % N == N - 1 else i + 1
            transition_func[i, 3] = i if i / N >= N - 1 else i + N

        # now we define the reward function
        reward_function[N ** 2 - 1, 2] = 1
        reward_function[N ** 2 - 1, 3] = 1

        super().__init__(name, num_states, action_space, state_space, transition_func, reward_function, starting_state)

    def get_name(self):
        return "grid_world"
