# MIT License
#
# Copyright (c) 2017 Markus Semmler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from plots.ManagedPlot import ManagedPlot


class MultiDimensionalHeatmap(ManagedPlot):

    def __init__(self, unique_name: str, num_densities: int, dims: list, height_width_ratio: float, color: str):
        """Creates a new DensityModelPlot. Therefore you have to specify
        how much different densities the plot should handle.

        Args:
            unique_name: The unique identifier
            config:
                num_densities: The number of densities to show.
                color: Specify the color, which should be used for
                height_width_ratio: Ratio for the layout creation process
                dims: The dimension of the input tensor
        """
        super().__init__(unique_name, [1, num_densities])
        height_width_ratio = height_width_ratio * num_densities
        self.dims = dims
        self.color = color

        # calculate the standard layout
        self.borders = [2, 1, 0, 0]
        self.layout, self.size, self.border_size = MultiDimensionalHeatmap.calc_layout(height_width_ratio, dims, self.borders)

    def plot(self, lst_tensor):

        dims = self.dims
        n = len(dims)

        assert len(lst_tensor) == len(self.plot_list)
        for k in range(len(lst_tensor)):

            current_tensor = lst_tensor[k][:dims[0], :dims[1], :dims[2], :dims[3]]
            maximum_per_sa = np.expand_dims(np.expand_dims(np.max(np.max(current_tensor, axis=3), axis=2), axis=2), axis=3)
            overall_maximum = np.max(maximum_per_sa)

            # convert the tensor
            self.tensor2d = MultiDimensionalHeatmap.apply_layout(current_tensor, dims, self.layout, self.size)
            self.plot_list[k].cla()
            self.plot_list[k].imshow(self.tensor2d, interpolation='nearest', cmap=plt.get_cmap(self.color))
            self.plot_list[k].set_xlim([-0.5, self.size[1] - 0.5])
            self.plot_list[k].set_ylim([-0.5, self.size[0] - 0.5])

        self.__plot_lines(self.layout, True)
        self.__plot_lines(self.layout, False)

    def __plot_lines(self, mask, horizontal):

        n = len(self.dims)
        dims = self.dims
        length = np.prod(np.power(dims, 1 - np.asarray(mask)))

        # find start index
        si = n - 1
        while self.layout[si] == (1 if horizontal else 0) and si > -1:
            si -= 1

        length /= self.dims[si]
        length -= 1
        length = int(length)

        if si != -1:
            for k in range(self.dims[si], (length + 1) * self.dims[si], self.dims[si]):
                for p in range(len(self.plot_list)):
                    self.plot_list[p].axhline(k - 0.5, color='black') if horizontal else \
                        self.plot_list[p].axvline(k - 0.5, color='black')

    @staticmethod
    def apply_layout(tensor, dims, layout, size):
        """This method converts the passed tensor into a 2D-representation.

        Args:
            tensor: A tensor of size len(self.dims)

        """
        layout = layout
        n = len(dims)

        # Reshape the tensor according to the layout
        slst = list()
        rlst = list()
        for i in reversed(range(n)):
            (slst if layout[i] == 0 else rlst).insert(0, i)

        shaped_tensor = tensor.transpose(slst + rlst).reshape(size)
        return shaped_tensor

    @staticmethod
    def calc_layout(height_width_ratio, dims, borders):
        """This method calculates the layout for the displaying, such
        that it gets optimally displayed.
        """

        def nelements(l, dir):

            ne = np.prod(dims ** np.equal(l, dir).astype(np.int32))
            nb_per_dimension = np.multiply(np.subtract(dims, 1), borders)
            cum_dims = np.cumprod(dims ** np.equal(l, dir))
            cum_dims[1:] = cum_dims[:-1]
            cum_dims[0] = 1
            border = np.sum(np.equal(l, dir) * nb_per_dimension * cum_dims)
            return ne, border

        def p(k, num, max_length):
            overall_length = k * num
            return overall_length if overall_length <= max_length else 0

        def h(l, k, height_width_ratio):
            return p(k, nelements(l, 0)[0], height_width_ratio) * p(k, nelements(l, 1)[0], 1)

        # get number of combinations and create corresponding numpy
        # vector and fill with the best values
        n = len(dims)
        best_value = 0
        best_size = (0, 0)
        best_border = (0, 0)
        best_l = np.zeros(n)
        current_l = np.ones(n)

        # try all combinations
        for i in range(2 ** 2):

            # one step forward for the combinations
            for k in reversed(range(0, n, 2)):
                if current_l[k] == 0:
                    current_l[k] += 1
                    current_l[k + 1] = 0
                    break

                current_l[k] = 0
                current_l[k + 1] = 1

            # get the number of elements in each direction
            nuh, hb = nelements(current_l, 0)
            nuw, hw = nelements(current_l, 1)
            nh = nuh  # + hb
            nw = nuw  # + hw
            k = np.minimum(1 / nw, height_width_ratio / nh)
            new_value = h(current_l, k, height_width_ratio)

            # save best value
            if new_value > best_value:
                best_value = new_value
                best_l = np.copy(current_l)
                best_size = (nuh, nuw)
                best_border = (hb, hw)

        return best_l.astype(np.int32), best_size, best_border