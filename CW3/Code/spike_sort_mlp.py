import numpy as np
from scipy.io import savemat

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import adam_v2

from spikedata import SpikeData


class SpikeSorting:

    def __init__(self):
  
        # Build keras MLP
        model = Sequential()
        model.add(Input(shape=41)) # Input layer
        model.add(Dense(20, activation='sigmoid')) # Hidden layer
        model.add(Dense(5, activation='softmax'))   # Output layer
        model.compile(loss='categorical_crossentropy', optimizer=adam_v2.Adam(learning_rate=0.01), metrics=['accuracy'])
        
        self.n = model
        

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
        self.n.fit(self.training_data.create_windows(), self.training_data.class_one_hot(), epochs=50, batch_size=16)



    def validate_mlp(self):     
        # Run spike detection and comparison on validation data
        spike_score = self.validation_data.compare_spikes()  

        # Classify detected spikes
        predicted = self.n.predict(self.validation_data.create_windows(), batch_size=16)
        # Convert probabilties back to class labels (1-5)
        predicted = np.argmax(predicted, axis=1) + 1

        # Compare to known classes
        classified = np.where(predicted == self.validation_data.classes)[0]

        # Score classifier method
        class_score = (len(classified) / len(self.validation_data.spikes)) * 100

        # Performance metrics
        print(f'Spike detection score: {spike_score:.1f}')
        print(f'Class detection score: {class_score:.1f}')
        print(f'Overall score:{(spike_score*class_score)/100:.1f}')

        print('Real Class Breakdown:')
        self.class_breakdown(self.validation_data.classes)
        print('Predicted Class Breakdown')
        self.class_breakdown(predicted)

        cm = confusion_matrix(self.validation_data.classes, predicted)
        print(cm)
        cr = classification_report(self.validation_data.classes, predicted)
        print(cr)




    def submission(self):
        self.submission_data = SpikeData()
        self.submission_data.load_mat('submission.mat')

        self.submission_data.filter_data([25,1900], 'band')
        #self.submission_data.plot_data(0, len(self.submission_data.data))

        self.submission_data.detect_spikes()

        predicted = self.n.predict(self.submission_data.create_windows(), batch_size=16)      
        predicted = np.argmax(predicted, axis=1) + 1 

        print('Class Breakdown')
        self.class_breakdown(predicted)

        mat_file = {'Index': self.submission_data.spikes, 'Class':predicted}
        savemat('13772.mat', mat_file)


    def class_breakdown(self, classes):
        unique, counts = np.unique(classes, return_counts=True)
        breakdown = dict(zip(unique, counts))

        for key, val in breakdown.items():
            print(f'Type {key}: {val}')


        







if __name__ == '__main__':

    s = SpikeSorting()
    s.train_mlp()
    s.validate_mlp()
    s.submission()


    



