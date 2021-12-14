from scipy.signal import find_peaks, butter, sosfiltfilt
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

#SpikeData - store and maniuplate spike data, as well as spike index and class data
class SpikeData:

    # Default constructor
    def __init__(self, data=None, spikes=None, classes=None):
        self.data = data
        self.spikes = spikes
        self.classes = classes

    # Load mat file
    def load_mat(self, file, train=False):
        data_set = loadmat(file, squeeze_me=True)

        self.data = data_set['d']

        # If training, also load known index and class data
        if train:        
            self.spikes = data_set['Index']
            self.classes = data_set['Class']

    # Plots data
    def plot_data(self, x, xlen):
        plt.plot(range(x, x+xlen), self.data[x:x+xlen])
        plt.show()

    # Sort index and class data by index, simultaneously
    def sort(self):        
        self.spikes, self.classes = map(np.array, zip(*sorted(zip(self.spikes, self.classes))))   

    # Apply filter to recording data
    def filter_data(self, cutoff, type, fs=25e3, order=2):
        # Low pass butterworth filter
        sos = butter(order, cutoff, btype=type, output='sos', fs=fs)
        filtered = sosfiltfilt(sos, self.data)
        self.data = filtered   
    
    # Spilt one spike data set into 2 (training/validation)
    def split(self, percent):
        # Get index to split at
        idx = int(len(self.data)*(1.0-percent))
        # Slice data
        split_data = self.data[idx:]
        self.data = self.data[:idx]

        # Slice spike and class index data
        i = int(np.argmax(self.spikes >= idx))        
        split_spikes = self.spikes[i:] - idx
        self.spikes = self.spikes[:i]
        split_classes = self.classes[i:]
        self.classes = self.classes[:i]        

        # Construct new SpikeData object with sliced data and return
        return SpikeData(split_data, split_spikes, split_classes)


    # Detect spike using standard signal processing techniques    
    def detect_spikes(self):
        # set peak threshold at 5 median absolute distributions of the data 
        mad = 5 * np.median(np.abs(self.data)/0.6745) 
        # Detect peaks 
        peaks, _ = find_peaks(self.data, height=mad) 

        # Overwrite class variable
        self.spikes = peaks
        return peaks

    # If training data, compare spike detection method to known index/class vectors
    def compare_spikes(self, tol=25):
        # Get known spike indexes   
        known = self.spikes

        # Overwrite with detected spikes
        detected = self.detect_spikes()

        # Initialise arrays
        spikes = np.zeros(len(detected))
        classes = np.zeros(len(detected))

        # Loop through detected spikes
        for i, spike in enumerate(detected):

            # Get all matching indexes in a tolerance
            found = np.where((known > spike-tol-15) & (known < spike+tol))[0]

            if len(found) > 0:
                # Mark as correct detection
                spikes[i] = spike

                # Sometimes a detected spike could correspond with multiple known spikes
                # Select the closest one
                diff = abs(known[found] - spike)
                idx = found[np.argmin(diff)]

                if idx < len(self.classes):
                    classes[i] = self.classes[idx]

                # Remove from selection, prevent duplicates
                known[idx] = 0

        # Remove all undetected spikes from the array, and their corresponding classes
        spikes = spikes[spikes != 0]
        classes = classes[classes != 0]

        # Overwrite class variables with new data
        self.spikes = spikes
        self.classes = classes

        # Score spike detecttion
        spike_score = (len(spikes) / len(known))
        return spike_score


    # Create window of data points around peak location
    def create_windows(self, window_size=41, offset=15):
        windows = np.zeros((len(self.spikes), window_size))

        # Loop through each spike index
        for i, spike in enumerate(self.spikes):       

            # Slice data array to get window     
            data = self.data[int(spike-offset):int(spike+window_size-offset)]

            windows[i, :] = data
        return windows

    


        
