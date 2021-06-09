from gerel.genome.factories import dense
from gerel.model.model import Model
from gerel.util.activations import build_sigmoid
from src.environment.walking_env import WalkingEnv
from src.training.mappings import action_map
import matplotlib.pyplot as plt

from gerel.util.datastore import DataStore
from gerel.genome.factories import from_genes
import os

import numpy as np

from src.params import STEPS, ENV_NAME, STATE_DIMS, ACTION_DIMS, LAYER_DIMS, DIR, \
    WEIGHT_LOW, WEIGHT_HIGH


def get_state_set(steps=STEPS, model=None):
    env = WalkingEnv(ENV_NAME, var=0)
    state = env.current_state
    i = 0
    states = []
    actions = []
    while i < steps:
        i += 1
        action = model(state)
        action = action_map(action)
        actions.append(action)
        env.take_action(action)
        env.step()
        state, _, _, _ = env.get_state()
        states.append(state)

    return states, actions


def get_model():
    genome = dense(
        input_size=STATE_DIMS,
        output_size=ACTION_DIMS,
        layer_dims=LAYER_DIMS,
        weight_low=WEIGHT_LOW,
        weight_high=WEIGHT_HIGH
    )
    sigmoid = build_sigmoid(c=10)
    model = Model(genome.to_reduced_repr, activation=sigmoid)
    return model


def load_model():
    generation = max([int(i) for i in os.listdir(DIR)])

    ds = DataStore(name=DIR)
    data = ds.load(generation)
    nodes, edges = data['best_genome']
    nodes = [n for n in nodes if n[4] == 'hidden']
    genome = from_genes(
        nodes, edges,
        input_size=STATE_DIMS,
        output_size=ACTION_DIMS,
        depth=len(LAYER_DIMS))
    sigmoid = build_sigmoid(c=10)
    return Model(genome.to_reduced_repr, activation=sigmoid)


if __name__ == '__main__':
    # model = get_model()
    model = load_model()
    states, actions = get_state_set(100, model)
    states = np.array(states)
    actions = np.array(actions)

    # for i in range(len(states[0])):
    #     plt.plot(states[:, i], color='red')

    for i in range(len(actions[0][0:5])):
        plt.plot(actions[:, i])

    plt.show()
