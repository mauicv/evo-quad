import numpy as np

STATE_DIMS = 51
ACTION_DIMS = 12
MIN_ACTION = -0.785398
MAX_ACTION = 0.785398

ACTION_RANGE = (MAX_ACTION - MIN_ACTION) / 2


def action_map(x):
    return  ACTION_RANGE * np.tanh(np.array(x))  # noqa
