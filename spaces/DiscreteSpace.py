import numpy as np


class DiscreteSpace:
    """This class represents a discrete space."""

    def __init__(self, N):
        """Constructs a discrete space.

        Args:
            N: The size of the space.
        """
        self.N = N

    def contains(self, x):
        """This method checks whether the passed element
        is contained inside of this space.

        Args:
            x: The element to check, if it lies in the space

        Returns:
            True, iff x is contained in this space.
        """
        return 0 <= x < self.N

    def get_size(self):
        """This method should deliver the size of this space.

        Returns:
            The size of the space"""
        return self.N

    def get_log2_size(self):
        """This method delivers the size of a binary vector which could
        encode all discrete values.

        Returns:
            The minimal size of the binary vector.
        """
        return int(np.ceil(np.log2(self.get_size())))

