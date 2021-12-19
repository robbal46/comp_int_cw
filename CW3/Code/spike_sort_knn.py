import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier

from spikedata import SpikeData


class SpikeSortingKNN:

    def __init__(self, k=5, p=2, verbose=True):        
  
       self.n = KNeighborsClassifier(n_neighbors=k, p=p)

       self.verbose = verbose
        

    def train_knn(self):       

        # Create SpikeData object for training data
        self.training_data = SpikeData()

        # Load in the training data set
        self.training_data.load_mat('training.mat', train=True)  

        # Sort index/class data 
        self.training_data.sort()

        # Take last 20% of training data set and use as a validation data set
        self.validation_data = self.training_data.split(0.2)

        # Filter the raw data
        self.training_data.filter_data(2500, 'low') 
        self.validation_data.filter_data(2500, 'low')

        # Run spike detection and comparison on training data
        self.training_data.compare_spikes()

        # Train the MLP with training dataset classes        
        self.n.fit(self.training_data.create_windows(), self.training_data.classes)



    def validate_knn(self):     
        # Run spike detection and comparison on validation data
        spike_score = self.validation_data.compare_spikes()  

        # Classify detected spikes
        predicted = self.n.predict(self.validation_data.create_windows())
        # Convert probabilties back to class labels (1-5)

        # Compare to known classes
        classified = np.where(predicted == self.validation_data.classes)[0]

        # Score classifier method
        class_score = (len(classified) / len(self.validation_data.spikes))

        #Performance metrics
        if self.verbose:
            print(f'Spike detection score: {spike_score:.4f}')
            print(f'Class detection score: {class_score:.4f}')
            print(f'Overall score:{(spike_score*class_score):.4f}')

            cm = confusion_matrix(self.validation_data.classes, predicted)
            print(cm)
            cr = classification_report(self.validation_data.classes, predicted, digits=4)
            print(cr)

            # # Plot any misclassified spikes
            # incorrect = np.where(predicted != self.validation_data.classes)[0]         
            # bd = list(self.class_breakdown(self.validation_data.classes[incorrect]))
            # fig, axs = plt.subplots(1,len(bd))
            # axs = np.array(axs).reshape(-1)
            # colours = ['c', 'r', 'g', 'b', 'm']
            # for i in incorrect:
            #     idx = int(self.validation_data.spikes[i])            
            #     c = int(self.validation_data.classes[i])            
            #     axs[bd.index(c)].plot(self.validation_data.data[idx-15:idx+26], colours[c-1])      
            # for j, ax in enumerate(axs):
            #     ax.set_title(f'Class {bd[j]:g}')
            # plt.tight_layout()
            # plt.subplots_adjust(wspace=0.1)
            # plt.show()

        return class_score


    # Run classifier on submission data set and create submission file
    def submission(self):
        self.submission_data = SpikeData()
        self.submission_data.load_mat('submission.mat')

        # Filter data with band pass as data is very noisy
        self.submission_data.filter_data([25,1800], 'band')

        spikes = self.submission_data.detect_spikes()
        print(f'{len(spikes)} spikes detected')

        predicted = self.n.predict(self.submission_data.create_windows())
        self.submission_data.classes = predicted      

        print('Class Breakdown')
        self.class_breakdown(predicted)

        mat_file = {'Index': self.submission_data.spikes, 'Class':predicted}
        savemat('13772.mat', mat_file)


    def class_breakdown(self, classes):
        unique, counts = np.unique(classes, return_counts=True)
        breakdown = dict(zip(unique, counts))

        for key, val in breakdown.items():
            print(f'Type {key:g}: {val}')
        return breakdown


if __name__ == '__main__':

    s = SpikeSortingKNN(1,2, verbose=True)
    s.train_knn()
    s.validate_knn()
    s.submission()
