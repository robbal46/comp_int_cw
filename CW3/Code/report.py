import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram

from spikedata import SpikeData
from spike_sort_knn import SpikeSortingKNN


def plots():
    training_data = SpikeData()
    training_data.load_mat('training.mat', train=True)
    training_data.sort()
    training_data.plot_data(0,1440000)

    training_data = SpikeData()
    training_data.load_mat('submission.mat')
    training_data.sort()
    training_data.plot_data(0,1440000)


def dist():
    training_data = SpikeData()
    training_data.load_mat('training.mat', train=True)
    training_data.sort()
    s = SpikeSortingKNN()
    s.class_breakdown(training_data.classes)




# for i in range(1,6):
#     cls = np.where(training_data.classes == i)[0][0]
#     idx = training_data.spikes[cls]

#     training_data.plot_data(idx, 50)

def freq():
    for f in range(1000, 4500, 500):
        data = SpikeData()
        data.load_mat('training.mat', train=True)
        data.sort()

        data.filter_data(f, 'low')
        actual_spikes = data.spikes
        acc = data.compare_spikes()

        print(f'{f}hz: {acc*100} accuracy')
    


dist()