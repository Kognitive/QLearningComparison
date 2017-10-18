import numpy as np
import math
import matplotlib.pyplot as plt

# do the discretization
discretizationp = 1500
discretizationv = 1000

# min and max values
min_position = -1.2
max_position = 0.6
max_speed = 0.07
goal_position = 0.5

# get intervals low and high
low = np.array([min_position, -max_speed])
high = np.array([max_position, max_speed])

# determine the step width
pos_step = (max_position - min_position) / (discretizationp - 1)
vel_step = (2 * max_speed) / (discretizationv - 1)
print("pos_step is {}".format(pos_step))
print("vel_step is {}".format(vel_step))

# build the transition matrix
trans = np.zeros([discretizationp, discretizationv, 3, 3])
rews = np.ones([discretizationp, discretizationv, 3, 1]) * -1

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
print("Filling Transition")
p_ticks = np.arange(min_position, max_position + pos_step / 100, pos_step)
v_ticks = np.arange(-max_speed, max_speed + vel_step / 100, vel_step)
for pi in range(len(p_ticks)):
    print(pi)
    for vi in range(len(v_ticks)):
        p = p_ticks[pi]
        v = v_ticks[vi]

        for a in range(3):
            next, reward, done, _ = step((p, v), a)
            pf = int((next[0] - min_position - pos_step / 100) / pos_step)
            ps = int((next[1] + max_speed - vel_step / 100) / vel_step)
            trans[pi, vi, a, :] = np.array([pf, ps, done])

print("Filled Transition")
# init q function
q_shape = (discretizationp, discretizationv, 3)
q_function = -np.zeros(q_shape)
next_q_function = -np.ones(q_shape) * 100
discount = 0.99

# repeat until converged
while np.max(np.abs(q_function - next_q_function)) >= 0.00001:

    print(np.max(np.abs(q_function - next_q_function)))

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
                bootstrapped_q_function[pi, vi, a] = rews[pi, vi, a] + discount * (np.max(next_q) if not next[2] else 0)

    # update the q function correctly
    next_q_function = np.squeeze(rews) + discount * np.squeeze(bootstrapped_q_function)


box = [min_position, max_position, - max_speed, max_speed]

min_position = -1.2
max_position = 0.6
max_speed = 0.07

fig = plt.figure(1)
fig.set_size_inches(6.2, 4.2)
ax1 = plt.axes([0.1, 0.1, 0.8, 0.8])
vf2 = ax1.imshow(np.transpose(np.max(next_q_function, axis=2)), interpolation='nearest', extent=box, aspect='auto')
plt.colorbar(vf2, ax=ax1)
plt.show()

# plot a different plot
fig = plt.figure(2)
fig.set_size_inches(6.2, 4.2)
plt.clf()
act_cmap = plt.cm.get_cmap('plasma', 3)

# print both plots
ba = plt.imshow(np.transpose(np.argmax(next_q_function, axis=2)), interpolation='nearest', cmap=act_cmap, vmin=-0.5, vmax=2.5,
                            extent=box, aspect='auto')

plt.xlabel("x")
plt.ylabel("v")
ba_cbar = plt.colorbar(ba, ticks=[0, 1, 2])
ba_cbar.set_ticklabels(['Left', 'Nothing', 'Right'])
plt.show()