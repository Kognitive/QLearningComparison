import numpy as np
import matplotlib.pyplot as plt
from environments.GridWorld import GridWorld

N = 10
env = GridWorld("grid", [1], N)
optimal_ih_rew, min_q, max_q, q_function = env.get_optimal(200, 0.99)

v_function = np.max(q_function, axis=1)
shaped_v_function = np.reshape(v_function, [N, N])

fig = plt.figure(100)
plt.imshow(shaped_v_function, interpolation='nearest')

# draw grid of black lines
for i in range(1, N):
    plt.axhline(y=i-0.5, xmin=-0.5, xmax=9.5, color='black', linewidth=0.5)
    plt.axvline(x=i-0.5, ymin=-0.5, ymax=9.5, color='black', linewidth=0.5)

plt.show()