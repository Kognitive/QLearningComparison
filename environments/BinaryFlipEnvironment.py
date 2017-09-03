# import the necessary packages

from src.environment.DiscreteEnvironment import DiscreteEnvironment
from src.util.spaces.DiscreteSpace import DiscreteSpace


class BinaryFlipEnvironment(DiscreteEnvironment):
    """This class represents a BinaryFlipEnvironment. It it is a simple toy
    example which can be used to test the learning algorithm."""

    def __init__(self, num_states, N):
        """Construct a new environment, you have to only specify the size N."""
        # set initial state and save sizes
        action_space = DiscreteSpace(N)
        state_space = DiscreteSpace(2 ** N)
        super().__init__(num_states, state_space, action_space, 0)

    def get_optimal(self, steps):

        N = self.AS.get_size()

        env_inf = BinaryFlipEnvironment(1, N)
        env_inf.reset()

        optimal_ih_rew = 0
        optimal_fh_rew = 0

        # Repeat for the number of steps
        k = 0
        while k < steps:

            # the agent should choose an action
            remaining_steps = steps - k

            # when evaluation steps are appropriate
            if remaining_steps >= 2 * N:
                for k_sub in range(2 * N):
                    optimal_ih_rew += env_inf.action(k_sub % N)
                    optimal_fh_rew += env_fin.action(k_sub % N)

                k += 2 * N

            else:

                # when the number is not equal
                if remaining_steps % 2 == 1:
                    optimal_fh_rew += env_fin.action(0)

                # integral, because remaining_steps is even
                end = int((remaining_steps - 1) / 2)
                for repeat in range(2):
                    for k_sub in range(N - end - 1, N):
                        optimal_fh_rew += env_fin.action(k_sub % N)

                for k_sub in range(remaining_steps):
                    optimal_ih_rew += env_inf.action(k_sub % N)

                k += remaining_steps

        # for this case
        if steps < 2 * N: optimal_ih_rew = optimal_fh_rew
        return optimal_ih_rew, optimal_fh_rew

    # basically flip the bit at the k-th position
    def perform_action(self, k):

        # check if it zero
        if self.value & (0x00000001 << k) > 0:

            # flip to one

            mask = 0xFFFFFFFE if k == 0 else BinaryFlipEnvironment.__rotl(0xFFFFFFFE, k, self.get_state_space().get_size())
            self.value = self.value & mask

            # define the reward
            reward = self.value

        else:

            # get decimal value afterwards
            reward = -self.value

            # flip to zero
            self.value |= 0x00000001 << k

        # print(self.state)

        # get decimal value afterwards
        return reward

    @staticmethod
    def __rotl(val, r_bits, max_bits):
        return (val << r_bits % max_bits) & (2 ** max_bits - 1) | \
            ((val & (2 ** max_bits - 1)) >> (max_bits - (r_bits % max_bits)))

    # this method can be used to print the state.
    def print_state(self):

        # simply print the state
        print(self.value)