import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from scipy.io import savemat

from spikedata import SpikeData


class SpikeSorting:

    def __init__(self):
        self.n = MLPClassifier(
            hidden_layer_sizes=(300,),
            random_state=1,
            max_iter=1000,
            activation='logistic',
            solver='adam',
            learning_rate_init=0.01,
            verbose=False
        )

        
        

    def train_mlp(self):
        # Create SpikeData object
        self.training_data = SpikeData()
        # Load in the training data set
        self.training_data.load_mat('training.mat', train=True)
        # Sort Index/Class vectors into ascending order
        self.training_data.sort()

        # Take last 10% of training data set and use as a validation data set
        self.validation_data = self.training_data.split(0.1)

        # Filter the raw data
        self.training_data.filter_data(25e3)
        self.validation_data.filter_data(25e3)

        # Train the MLP with training dataset classes
        self.n.fit(self.training_data.create_windows(), self.training_data.classes)



    def validate_mlp(self, tol=25):        
        known = self.validation_data.spikes
        detected = self.validation_data.detect_spikes()

        correct = np.zeros(len(detected))
        classes = np.zeros(len(detected))

        # Loop through detected spikes
        # Check if it matches an index within the known spikes (within tolerance)
        # If so, mark as a correct detection
        for i, spike in enumerate(detected):
            lower = spike - tol
            upper = spike + tol

            found = np.where((known > lower) & (known < upper))[0]

            if len(found) > 0:
                correct[i] = spike

                # Sometimes a detected spike could correspond with multiple known spikes
                # Select the closest one
                diff = abs(known[found] - spike)
                idx = found[np.argmin(diff)]
                classes[i] = self.validation_data.classes[idx]

        # Remove all undetected spikes from the array, and their corresponding classes
        correct = correct[correct != 0]
        classes = classes[classes != 0]

        # Score the spike detection method  
        n_total = len(known)
        n_correct = len(correct)
        spike_score = (n_correct / n_total) * 100


        # Classify detected spikes
        predicted = self.n.predict(self.validation_data.create_windows())
        # Compare to known classes
        classified = np.where(predicted == classes)[0]

        # Score classifier method
        n_classified = len(classified)
        class_score = (n_classified / n_correct) * 100

        print(f'From {n_total} spikes, {n_correct} were detected ({spike_score:.2f}%) successfully. ' 
            f'Of these, {n_classified} ({class_score:.2f}%) were classified correctly. '
            f'This gives a total success rate of {(spike_score*class_score)/100:.2f}%.')

        return spike_score, class_score 


    def submission(self):
        self.submission_data = SpikeData()
        self.submission_data.load_mat('submission.mat', train=False)

        self.submission_data.filter_data(25e3)

        self.submission_data.detect_spikes()

        predicted = self.n.predict(self.submission_data.create_windows()) 
        self.submission_data.classes = predicted       

        unique, counts = np.unique(predicted, return_counts=True)
        breakdown = dict(zip(unique, counts))

        print(f'Detected {len(self.submission_data.spikes)} spikes. '
            f'Class breakdown is - Type 1: {breakdown[1]}, Type 2: {breakdown[2]}, '
            f'Type 3: {breakdown[3]}, Type 4: {breakdown[4]}'
        )


        mat_file = {'Index': self.submission_data.spikes, 'Class':self.submission_data.classes}
        savemat('13772.mat', mat_file)

        







if __name__ == '__main__':

    s = SpikeSorting()
    s.train_mlp()
    s.validate_mlp()
    #s.submission()


    




