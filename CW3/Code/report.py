import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, welch

from spikedata import SpikeData
from spike_sort_knn import SpikeSortingKNN


def plots():
    training_data = SpikeData()
    training_data.load_mat('training.mat', train=True)
    training_data.sort()
    training_data.plot_data(0,1440000)
    training_data.plot_data(1000,500)
    training_data.filter_data(2500, 'low')
    training_data.plot_data(1000,500)

    submission_data = SpikeData()
    submission_data.load_mat('submission.mat')
    submission_data.plot_data(0,1440000)
    submission_data.filter_data([25,1900], 'band')
    submission_data.plot_data(0,1440000)

def classes():
    training_data = SpikeData()
    training_data.load_mat('training.mat', train=True)
    training_data.sort()
    training_data.filter_data(2500, 'low')
    fig, axs = plt.subplots(1,5)
    sample = [0,0,0,0,0]
    for i, spike in enumerate(training_data.spikes):
        idx = training_data.classes[i]-1
        if sample[idx] >= 10:
            axs[idx].plot(training_data.data[spike-15:spike+31])
            sample[idx] = 0
        else:
            sample[idx] += 1
    for j, ax in enumerate(axs):
        ax.set_title(f'Class {j+1}')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.show()


    s = SpikeSortingKNN(1,2)
    s.train_knn()
    s.submission()

    fig, axs = plt.subplots(1,5)
    sample = [0,0,0,0,0]
    for i, spike in enumerate(s.submission_data.spikes):
        idx = int(s.submission_data.classes[i]-1)
        if sample[idx] >= 10:
            axs[idx].plot(s.submission_data.data[spike-15:spike+31])
            sample[idx] = 0
        else:
            sample[idx] += 1
    for j, ax in enumerate(axs):
        ax.set_title(f'Class {j+1}')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.show()



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


def power():
    training_data = SpikeData()
    training_data.load_mat('training.mat', train=True)

    f, pxx_den = welch(training_data.data, 25e3)
    plt.semilogy(f, pxx_den)
    plt.title('Welch Power Spectrum - Training Data')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Linear Spectrum')
    plt.show()

    submission_data = SpikeData()
    submission_data.load_mat('submission.mat', train=False)

    f, pxx_den = welch(submission_data.data, 25e3)
    plt.semilogy(f, pxx_den)
    plt.title('Welch Power Spectrum - Submission Data')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Linear Spectrum')
    plt.show()

#plots()
classes()
#power()