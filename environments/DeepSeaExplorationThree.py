import numpy as np

from environments.DeterministicMDP import DeterministicMDP
from spaces.DiscreteSpace import DiscreteSpace


class DeepSeaExplorationThree(DeterministicMDP):

    def __init__(self, name, num_states, N):

        # create the state and action space
        self.inner_size = N
        state_space = DiscreteSpace(N ** 2)
        action_space = DiscreteSpace(2)

        # get size of state and action space
        size_space = state_space.get_size()
        size_action = action_space.get_size()

        # one maps to 2
        starting_state = N + 1

        # specify the transition function
        transition_func = np.zeros((size_space, size_action), dtype=np.int32)
        reward_function = np.zeros((size_space, size_action), dtype=np.float64)

        # sample the left action
        left = 1
        right = 1 - left
        chest = 2 * 0 - 1

        # iterate over and fill with the transitions
        for x in range(N):
            for y in range(N):
                pos = y * N + x
                y = y if y == N - 1 else y + 1
                left_x = x if x == 0 else x - 1
                right_x = x if x == N - 1 else x + 1

                transition_func[pos, left] = y * N + left_x
                transition_func[pos, right] = y * N + right_x

        reward_function[N ** 2 - 1, right] = chest

        super().__init__(name, num_states, action_space, state_space, transition_func, reward_function, starting_state)

    def get_name(self):
        return "deep_sea_three"