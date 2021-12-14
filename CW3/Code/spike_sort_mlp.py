import numpy as np
from scipy.io import savemat

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.neural_network import MLPClassifier

from spikedata import SpikeData


class SpikeSortingMLP:

    def __init__(self):
  
        self.n = MLPClassifier(
            hidden_layer_sizes=(20,),
            random_state=1,
            max_iter=1000,
            activation='logistic',
            solver='adam',
            learning_rate_init=0.01,
            batch_size=16,
            verbose=False
        )        

    def train_mlp(self):       

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



    def validate_mlp(self):     
        # Run spike detection and comparison on validation data
        spike_score = self.validation_data.compare_spikes()  

        # Classify detected spikes
        predicted = self.n.predict(self.validation_data.create_windows())

        # Compare to known classes
        classified = np.where(predicted == self.validation_data.classes)[0]

        # Score classifier method
        class_score = (len(classified) / len(self.validation_data.spikes))

        # Performance metrics
        print(f'Spike detection score: {spike_score:.4f}')
        print(f'Class detection score: {class_score:.4f}')
        print(f'Overall score:{(spike_score*class_score):.4f}')

        print('Real Class Breakdown:')
        self.class_breakdown(self.validation_data.classes)
        print('Predicted Class Breakdown')
        self.class_breakdown(predicted)

        cm = confusion_matrix(self.validation_data.classes, predicted)
        print(cm)
        cr = classification_report(self.validation_data.classes, predicted, digits=4)
        print(cr)



    # This MLP based classification approach was not selected for optimization and
    # thus the submission file does not need to generated
    # However, uncomment this function should you wish to run the MLP classifier on
    # submission data

    # Creates the submission file
    # def submission(self):
    #     self.submission_data = SpikeData()
    #     self.submission_data.load_mat('submission.mat')

    #     self.submission_data.filter_data([25,1900], 'band')
    #     #self.submission_data.plot_data(0, len(self.submission_data.data))

    #     self.submission_data.detect_spikes()

    #     predicted = self.n.predict(self.submission_data.create_windows())      

    #     print('Class Breakdown')
    #     self.class_breakdown(predicted)

    #     mat_file = {'Index': self.submission_data.spikes, 'Class':predicted}
    #     savemat('13772.mat', mat_file)


    def class_breakdown(self, classes):
        unique, counts = np.unique(classes, return_counts=True)
        breakdown = dict(zip(unique, counts))

        for key, val in breakdown.items():
            print(f'Type {key}: {val}')



if __name__ == '__main__':

    s = SpikeSortingMLP()
    s.train_mlp()
    s.validate_mlp()
    #s.submission()


    



