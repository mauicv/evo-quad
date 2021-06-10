import numpy as np
from src.params import MIN_ACTION, MAX_ACTION

ACTION_RANGE = (MAX_ACTION - MIN_ACTION) / 2


def action_map(x):
    return  ACTION_RANGE * np.tanh(np.array(x))  # noqa
