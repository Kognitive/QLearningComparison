import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from environments.GridWorld import GridWorld
from environments.ExplorationChain import ExplorationChain
from environments.BinaryFlipEnvironment import BinaryFlipEnvironment
from environments.DeepSeaExploration import DeepSeaExploration
from environments.DeepSeaExplorationTwo import DeepSeaExplorationTwo

env = "binflip"
if env == "grid":

    N = 10
    env = GridWorld("grid", [1], N)
    optimal_ih_rew, minimal_ih_rew, min_q, max_q, q_function = env.get_optimal(200, 0.99)

    v_function = np.max(q_function, axis=1)
    shaped_v_function = np.reshape(v_function, [N, N])

    fig = plt.figure(100)
    plt.imshow(shaped_v_function, interpolation='nearest')

    # draw grid of black lines
    for i in range(1, N):
        plt.axhline(y=i-0.5, xmin=-0.5, xmax=9.5, color='black', linewidth=0.5)
        plt.axvline(x=i-0.5, ymin=-0.5, ymax=9.5, color='black', linewidth=0.5)

    ax = plt.gca()
    plt.xticks(np.arange(0, 9 + 1, 1.0), ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10"))
    plt.yticks(np.arange(0, 9 + 1, 1.0), ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10"))

    plt.show()

if env == "expchain":

    N = 10
    env = ExplorationChain("grid", [1], N)
    optimal_ih_rew, _, min_q, max_q, q_function = env.get_optimal(200, 0.99)

    v_function = np.max(q_function, axis=1)
    shaped_v_function = np.expand_dims(v_function, 0)

    fig = plt.figure(100)
    plt.imshow(shaped_v_function, interpolation='nearest')

    # draw grid of black lines
    for i in range(1, N):
        plt.axvline(x=i-0.5, ymin=-0.5, ymax=19.5, color='black', linewidth=0.5)

    plt.xticks(np.arange(0, 9 + 1, 1.0), ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10"))
    plt.yticks(np.arange(0, 0 + 1, 1.0), (""))
    plt.show()

if env == "deepsea":

    N = 10
    env = DeepSeaExplorationTwo("deep", [1], N)
    optimal_ih_rew, _, min_q, max_q, q_function = env.get_optimal(200, 0.99)

    v_function = np.max(q_function, axis=1)
    shaped_v_function = np.reshape(v_function, [N, N])

    fig = plt.figure(100)
    plt.imshow(shaped_v_function, interpolation='nearest')

    for i in range(1, N):
        plt.axhline(y=i - 0.5, xmin=0, xmax=(i+1)/N, color='black', linewidth=0.5)
        plt.axvline(x=i - 0.5, ymin=0, ymax=(N - i + 1)/N, color='black', linewidth=0.5)
        # for i in range(1, N):
        # plt.axvline(x=i - 0.5, ymin=-0.5, ymax=i-0.5, color='black', linewidth=0.5)

    for x in np.arange(0.5, 10.5, 1.0):
        for y in np.arange(-0.5, x, 1.0):
            ax = plt.gca()
            ax.add_patch(
                patches.Rectangle(
                    (x, y),
                    1,
                    1,
                    fill='white',  # remove background
                    alpha=1,
                    facecolor='white'
                )
            )

if env == "binflip":

    N = 4
    env = BinaryFlipEnvironment("grid", [1], N)
    optimal_ih_rew, _, min_q, max_q, q_function = env.get_optimal(200, 0.99)

    v_function = np.max(q_function, axis=1)
    shaped_v_function = np.reshape(v_function, [N, N])

    fig = plt.figure(100)
    plt.imshow(shaped_v_function, interpolation='nearest')

    arr = shaped_v_function[:, 3]
    shaped_v_function[:, 3] = shaped_v_function[:, 2]
    shaped_v_function[:, 2] = arr

    arr = shaped_v_function[3, :]
    shaped_v_function[3, :] = shaped_v_function[2, :]
    shaped_v_function[2, :] = arr

    # draw grid of black lines
    for i in range(1, N):
        plt.axvline(x=i - 0.5, ymin=-0.5, ymax=19.5, color='black', linewidth=0.5)
        plt.axhline(y=i - 0.5, xmin=-0.5, xmax=19.5, color='black', linewidth=0.5)

    plt.xticks(np.arange(0, N, 1.0), ("00", "01", "11", "10"))
    plt.yticks(np.arange(0, N, 1.0), ("00", "01", "11", "10"))
    plt.show()