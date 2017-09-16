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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

class ManagedPlot:

    """This class can be used to give a class access to subplots managed
    by itself."""

    def __init__(self, unique_name, grid_shape):
        """Constructs a new LayoutManagedPlot. Creates subplots according to
        grid_shapes.
        """
        assert len(grid_shape) == 2
        self.grid_shape = grid_shape
        self.unique_name = unique_name
        self.figure = plt.figure(unique_name)

        # create the list of plots
        num_elements = np.prod(self.grid_shape)
        self.plot_list = num_elements * [None]

        # calculate the base number
        basenum = 100 * grid_shape[0] + 10 * grid_shape[1]

        for plt_ind in range(num_elements):
            self.plot_list[plt_ind] = self.figure.add_subplot(basenum + plt_ind + 1)

        self.writer = None

    def start_recording(self, agent_folder, fps):

        conc_folder = os.path.join(agent_folder, self.unique_name) + ".mp4"
        self.writer = animation.FFMpegFileWriter(fps=fps)
        self.writer.setup(self.figure, conc_folder, dpi=300, frame_prefix='run/tmp/_tmp_{}'.format(self.unique_name))

    def store_frame(self):
        self.writer.grab_frame()

    def stop_recording(self):
        self.writer.finish()

