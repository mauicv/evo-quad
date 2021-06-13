from src.environment.walking_env import WalkingEnv
from src.training.mappings import action_map
from src.params import STEPS, ENV_NAME
from src.util import load_model
# from src.util import get_model

import matplotlib.pyplot as plt
import numpy as np


def get_state_set(steps=STEPS, model=None):
    env = WalkingEnv(ENV_NAME, var=0)
    state = env.current_state
    i = 0
    states = []
    pre_actions = []
    actions = []
    while i < steps:
        i += 1
        action = model(state)
        pre_actions.append(action)
        action = action_map(action)
        actions.append(action)
        env.take_action(action)
        env.step()
        state, _, _, _ = env.get_state()
        states.append(state)

    return states, actions, pre_actions


if __name__ == '__main__':
    # model = get_model()
    model = load_model()
    states, actions, pre_actions = get_state_set(500, model)
    states = np.array(states)
    actions = np.array(actions)
    pre_actions = np.array(pre_actions)
    # for i in range(len(pre_actions[0])):
    #     plt.plot(pre_actions[:, i], color='red')

    for i in range(len(actions[0])):
        plt.plot(actions[:, i], color='blue')

    plt.show()
