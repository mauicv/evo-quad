import numpy as np


class Box:
    def __init__(self, shape, high, low):
        self.shape = shape
        self.high = high
        self.low = low

    def sample(self):
        return np.clip(np.random.normal(0, 0.01, size=(self.shape)), -1, 1) \
            * (self.high - self.low)

    @property
    def arc_sizes(self):
        return [high - low for low, high in zip(self.low, self.high)]
