from gerel.genome.factories import dense
from gerel.model.model import Model
from gerel.util.activations import build_leaky_relu
# import click
# import time
from src.environment.walking_env import WalkingEnv
from src.training.mappings import action_map
# from gerel.genome.factories import dense, from_genes
import matplotlib.pyplot as plt

import numpy as np


STATE_DIMS = 51
ACTION_DIMS = 12
MIN_ACTION = -0.785398
MAX_ACTION = 0.785398
LAYER_DIMS = [20, 20, 20]
ENV_NAME = 'walking-quadruped'
STEPS = 100


def get_state_set(steps=STEPS):
    env = WalkingEnv(ENV_NAME, var=0)
    state = env.current_state
    i = 0
    states = []
    while i < STEPS:
        i += 1
        action = np.random.uniform((ACTION_DIMS), -1, 1) * MAX_ACTION
        state, _, _, _ = env.step(action)
        states.append(state)
    return states


def get_model():
    genome = dense(
        input_size=STATE_DIMS,
        output_size=ACTION_DIMS,
        layer_dims=LAYER_DIMS,
        weight_low=-1,
        weight_high=1,
    )
    leaky_relu = build_leaky_relu()
    model = Model(genome.to_reduced_repr, activation=leaky_relu)
    return model


if __name__ == '__main__':
    model = get_model()
    states = get_state_set(100)
    states = np.array(states)
    for i in range(len(states[0])):
        plt.plot(states[:, i], color='red')

    actions = np.array([model(state) for state in states])/6
    for i in range(len(actions[0])):
        plt.plot(actions[:, i], color='blue')

    actions = np.array([action_map(action) for action in actions])
    for i in range(len(actions[0])):
        plt.plot(actions[:, i], color='blue')

    plt.show()
