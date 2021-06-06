import matplotlib.pyplot as plt
from gerel.util.datastore import DataStore
import os


DIR = './data/default/'

if __name__ == '__main__':
    if os.path.isdir(DIR) and os.listdir(DIR):
        ds = DataStore(name=DIR)
        means = []
        bests = []
        worsts = []
        for generation in ds.generations():
            data = ds.generations()
            means.append(generation['mean_fitness'])
            bests.append(generation['best_fitness'])
            worsts.append(generation['worst_fitness'])
        plt.plot(means)
        plt.plot(bests)
        plt.plot(worsts)
        plt.show()
    else:
        print('No training data present. Use train.py to train a solution.')
