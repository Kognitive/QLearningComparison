import numpy as np
import math
import matplotlib.pyplot as plt

# do the discretization
discretization = 10

# min and max values
min_position = -1.2
max_position = 0.6
max_speed = 0.07
goal_position = 0.5

# get intervals low and high
low = np.array([min_position, -max_speed])
high = np.array([max_position, max_speed])

# determine the step width
pos_step = (max_position - min_position) / discretization
vel_step = (2 * max_speed) / discretization
print("pos_step is {}".format(pos_step))
print("vel_step is {}".format(vel_step))

# build the transition matrix
trans = np.zeros([discretization, discretization, 3, 3])
rews = np.zeros([discretization, discretization, 3, 1]) * -1

def step(state, action):

    position, velocity = state
    velocity += (action - 1) * 0.001 + math.cos(3 * position) * (-0.0025)
    velocity = np.clip(velocity, -max_speed, max_speed)
    position += velocity
    position = np.clip(position, min_position, max_position)
    if (position == min_position and velocity < 0): velocity = 0

    done = bool(position >= goal_position)
    reward = -1.0

    state = (position, velocity)
    return np.array(state), reward, done, {}

p_ticks = np.arange(min_position, max_position, pos_step)
v_ticks = np.arange(-max_speed, max_speed, vel_step)
for pi in range(len(p_ticks)):
    for vi in range(len(v_ticks)):
        p = p_ticks[pi]
        v = v_ticks[vi]

        for a in range(3):
            next, reward, done, _ = step((p, v), a)
            pf = int((next[0] - min_position) / pos_step) - 1
            ps = int((next[1] + max_speed) / vel_step) - 1
            trans[pi, vi, a, :] = np.array([pf, ps, done])

# init q function
q_shape = (discretization, discretization, 3)
q_function = -np.ones(q_shape)
next_q_function = np.zeros(q_shape)
discount = 0.99

# repeat until converged
while np.max(np.abs(q_function - next_q_function)) >= 0.001:

    # create next bootstrapped q function
    q_function = next_q_function
    bootstrapped_q_function = np.empty(q_shape)

    # iterate over all fields
    for pi in range(len(p_ticks)):
        for vi in range(len(v_ticks)):
            p = p_ticks[pi]
            v = v_ticks[vi]

            for a in range(3):
                next = trans[pi, vi, a]
                next_q = q_function[int(next[0]), int(next[1]), :]
                bootstrapped_q_function[pi, vi, a] = rews[pi, vi, a] + discount * (np.max(next_q) if not done else 0)

    # update the q function correctly
    next_q_function = np.squeeze(rews) + discount * np.squeeze(bootstrapped_q_function)

fig = plt.figure()
plt.imshow(np.max(next_q_function, axis=2), interpolation='nearest')
plt.show()