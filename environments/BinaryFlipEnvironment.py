import numpy as np

from environments.DeterministicMDP import DeterministicMDP
from spaces.DiscreteSpace import DiscreteSpace


class BinaryFlipEnvironment(DeterministicMDP):

    def __init__(self, name, num_states, N):

        # create the state and action space
        self.inner_size = N
        state_space = DiscreteSpace(2 ** N)
        action_space = DiscreteSpace(N)

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
            for j in range(size_action):
                next_state = self.flip_bit(i, j, N)
                transition_func[i, j] = next_state
                reward_function[i, j] = np.sign(i - next_state) * np.minimum(i, next_state)

        super().__init__(name, num_states, action_space, state_space, transition_func, reward_function, starting_state)

    def flip_bit(self, number, N, bits):
        """This method flips a single bit"""
        if (number & (1 << N)) > 0:
            num = number & ~(1 << N)
        else:
            num = number | (1 << N)

        return int("{:0{}b}".format(num, bits), 2)

    def get_name(self):
        return "bin_flip"