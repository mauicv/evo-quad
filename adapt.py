from gerel.genome.factories import dense
from gerel.model.model import Model
from gerel.util.activations import build_leaky_relu, build_sigmoid
# import click
# import time
from src.environment.walking_env import WalkingEnv
from src.training.mappings import action_map
# from gerel.genome.factories import dense, from_genes
import matplotlib.pyplot as plt

from gerel.util.datastore import DataStore
from gerel.genome.factories import from_genes
import os

import numpy as np

DIR = './data/default/'
STATE_DIMS = 51
ACTION_DIMS = 12
MIN_ACTION = -0.785398
MAX_ACTION = 0.785398
LAYER_DIMS = [20, 20]
ENV_NAME = 'walking-quadruped'
STEPS = 100


def get_state_set(steps=STEPS, model=None):
    env = WalkingEnv(ENV_NAME, var=0)
    state = env.current_state
    i = 0
    states = []
    actions = []
    while i < STEPS:
        i += 1
        action = model(state)
        action = action_map(action)
        actions.append(action)
        state, _, _, _ = env.step(action)
        states.append(state)

    return states, actions


def get_model():
    genome = dense(
        input_size=STATE_DIMS,
        output_size=ACTION_DIMS,
        layer_dims=LAYER_DIMS,
        weight_low=-1,
        weight_high=1,
    )
    sigmoid = build_sigmoid(c=10)
    model = Model(genome.to_reduced_repr, activation=sigmoid)
    return model


def load_model():
    generation = max([int(i) for i in os.listdir(DIR)])

    ds = DataStore(name=DIR)
    data = ds.load(generation)
    nodes, edges = data['best_genome']
    input_num = len([n for n in nodes if n[4] == 'input'])
    output_num = len([n for n in nodes if n[4] == 'output'])
    nodes = [n for n in nodes if n[4] == 'hidden']
    genome = from_genes(
        nodes, edges,
        input_size=input_num,
        output_size=output_num,
        depth=len(LAYER_DIMS))
    sigmoid = build_sigmoid(c=10)
    return Model(genome.to_reduced_repr, activation=sigmoid)


if __name__ == '__main__':
    model = get_model()
    # model = load_model()
    states, actions = get_state_set(100, model)
    states = np.array(states)
    actions = np.array(actions)

    # for i in range(len(states[0])):
    #     plt.plot(states[:, i], color='red')

    for i in range(len(actions[0])):
        plt.plot(actions[:, i], color='blue')

    plt.show()
