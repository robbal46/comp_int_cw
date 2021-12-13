import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram

from spikedata import SpikeData


# Create SpikeData object
training_data = SpikeData()
# Load in the training data set
training_data.load_mat('training.mat', train=True)
# Sort Index/Class vectors into ascending order
training_data.sort()


# for i in range(1,6):
#     cls = np.where(training_data.classes == i)[0][0]
#     idx = training_data.spikes[cls]

#     training_data.plot_data(idx, 50)


training_data.filter_data()

f, pxx_den = periodogram(training_data.data, 25e3)
plt.semilogy(f, pxx_den)
plt.show()
