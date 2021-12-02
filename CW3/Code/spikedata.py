from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from scipy.io import loadmat

import numpy as np
import matplotlib.pyplot as plt

class SpikeData:

    def __init__(self, data=None, spikes=None, classes=None):
        self.data = data
        self.spikes = spikes
        self.classes = classes

        self.f_sample = 25e3

    
    def load_mat(self, file, train):
        data_set = loadmat(file, squeeze_me=True)

        self.data = data_set['d']

        if train:        
            self.spikes = data_set['Index']
            self.classes = data_set['Class']


    def plot_data(self):
        plt.plot(range(len(self.data)), self.data)
        plt.vlines(self.spikes[0], min(self.data), max(self.data), colors='r')
        plt.show()


    def sort(self):
        # Sort index and class data by index, simultaneously
        self.spikes, self.classes = map(np.array, zip(*sorted(zip(self.spikes, self.classes))))   


    def butter_band_pass_filter(self, data, f):
        nyq = f / 2
        lower = 30 / nyq
        upper = 3000 / nyq

        b,a = butter(4, [lower,upper], btype='band', output='ba')
        filtered = filtfilt(b, a, data)
        return filtered

    def filter_data(self, f):
        # First apply band pass butterworth filter
        nyq = f / 2
        lower = 30 / nyq
        upper = 3000 / nyq
        b,a = butter(4, [lower,upper], btype='band', output='ba')
        filtered = filtfilt(b, a, self.data)

        filtered = self.butter_band_pass_filter(self.data, f)
        filtered = savgol_filter(filtered, 65, 6)
        self.data = filtered
    

    def split(self, percent):
        # Get index to split at
        idx = int(len(self.data)*(1.0-percent))
        split_data = self.data[idx:]
        self.data = self.data[:idx]

        i = int(np.argmax(self.spikes >= idx))        
        split_spikes = self.spikes[i:] - idx
        self.spikes = self.spikes[:i]

        split_classes = self.classes[i:]
        self.classes = self.classes[:i]        

        return SpikeData(split_data, split_spikes, split_classes)

        
    def detect_spikes(self):
        # detect peaks that exceed std deviation of data
        std_dev = np.std(self.data)    
        peaks,_ = find_peaks(self.data, height=std_dev)    

        # find peaks returns the index of the peak amplitudes
        spikes = np.zeros(len(peaks))

        for i, index in enumerate(peaks):
            # Start of spike is where gradient is max
            grad = np.diff(self.data[index-15:index])

            # Number of indexes back from peak
            offset = 15 - np.argmax(grad)-1
            
            spikes[i] = index - offset     


        self.spikes = spikes
        return spikes




    def create_windows(self, window_size=100):
        before = int(window_size*0.25)
        after = int(window_size*0.75)

        windows = np.zeros((len(self.spikes), window_size))

        for i, spike in enumerate(self.spikes):            
            data = self.data[int(spike):int(spike+window_size)]

            min_val = np.min(data)
            max_val = np.max(data)
            data_normal = (data - min_val) / (max_val - min_val)

            windows[i, :] = data_normal

        return windows


        
