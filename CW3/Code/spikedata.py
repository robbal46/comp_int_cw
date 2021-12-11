from scipy.signal import find_peaks, savgol_filter, butter, filtfilt, sosfiltfilt
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


    def plot_data(self, x, xlen):
        plt.plot(range(x, x+xlen), self.data[x:x+xlen])
        plt.show()


    def sort(self):
        # Sort index and class data by index, simultaneously
        self.spikes, self.classes = map(np.array, zip(*sorted(zip(self.spikes, self.classes))))   


    def filter_data(self, f):
        sos = butter(1, 3000, btype='low', output='sos', fs=25e3)
        filtered = sosfiltfilt(sos, self.data)
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
        std_dev = 3 * np.std(self.data)   
        #mad = 6 * np.median(np.abs(self.data)/0.6745) 
        peaks,_ = find_peaks(self.data, height=std_dev)    

        # find peaks returns the index of the peak amplitudes
        spikes = np.zeros(len(peaks), dtype=int)

        for i, index in enumerate(peaks):
            # Start of spike is where gradient is max
            grad = np.diff(self.data[index-15:index])

            # Number of indexes back from peak
            offset = 15 - np.argmax(grad)-1
            
            spikes[i] = index - offset 

        self.spikes = spikes
        return spikes

    # Create array of data points around peak
    def create_windows(self, window_size=100):
        # Start/end of window
        before = 0
        after = int(window_size - before)

        windows = np.zeros((len(self.spikes), window_size))

        for i, spike in enumerate(self.spikes):            
            data = self.data[int(spike-before):int(spike+after)]

            min_val = np.min(data)
            max_val = np.max(data)
            # Normalize data
            data_normal = (data - min_val) / (max_val - min_val)

            windows[i, :] = data_normal

        return windows

    # Integer to one hot encoded vector for class labels
    def class_to_vector(self):
        
        c = int(max(self.classes))
        vec = self.classes - 1
        vec = vec.reshape(-1)
        one_hot = np.eye(c)[vec]

        return one_hot
    


        
