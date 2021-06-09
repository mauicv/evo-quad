import matplotlib.pyplot as plt
from gerel.util.datastore import DataStore
import os


DIR = './data/default/'


def moving_average(arr):
    N = 50
    cumsum, moving_aves = [0], []

    for i, x in enumerate(bests, 1):
        cumsum.append(cumsum[i-1] + x)
        if i >= N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            moving_aves.append(moving_ave)
    return moving_aves


if __name__ == '__main__':
    if os.path.isdir(DIR) and os.listdir(DIR):
        ds = DataStore(name=DIR)
        # means = []
        bests = []
        # worsts = []
        for generation in ds.generations():
            data = ds.generations()
            # means.append(generation['mean_fitness'])
            bests.append(generation['best_fitness'])
            # worsts.append(generation['worst_fitness'])
        # plt.plot(means)
        moving_aves = moving_average(bests)
        plt.plot(moving_aves)
        # plt.plot(bests)
        # plt.plot(worsts)
        plt.show()
    else:
        print('No training data present. Use train.py to train a solution.')
