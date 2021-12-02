from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pyplot as plt
import os

from neuralnetwork import NeuralNetwork


class ImageRecognition:

    def __init__(self, i, h, o, lr, verbose=True):

        self.i_nodes = i
        self.h_nodes = h
        self.o_nodes = o
        self.l_rate = lr

        self.n = NeuralNetwork([self.i_nodes, self.h_nodes, self.o_nodes], self.l_rate)

        self.score = 0

        # Debug using print statements
        self.verbose = verbose

    #Helper function for reading csv files
    def read_csv(self, filename):
        # Reads MNIST dataset from MNIST directory
        try:
            filepath = os.path.join(os.path.dirname(__file__), f'MNIST/{filename}.csv')
            file = open(filepath, 'r')
            data = file.readlines()
            file.close()
            return data

        except FileNotFoundError:
            print(f'Could not find file at {filepath}')
            exit()

    def update_lr(self, lr):
        self.l_rate = lr
        self.n.update_lr(self.l_rate)

    # Draws the MNIST data
    def draw(self, dataset):
        # Open the 100 training samples in read mode
        data_list = self.read_csv(dataset)
        # Take the first line (data_list index 0, the first sample), and split it up based on the commas
                
        # all_values now contains a list of [label, pixel 1, pixel 2, pixel 3, ... ,pixel 784]
        for data in data_list:
            try:
                all_values = data.split(',')

                values = np.asfarray(all_values[1:])
                # Take the long list of pixels (but not the label), and reshape them to a 2D array of pixels
                image_array = values.reshape((28, 28))
                # Plot this 2D array as an image, use the grey colour map and don’t interpolate
                fig, (ax1, ax2) = plt.subplots(1, 2)                
                im1 = ax1.imshow(image_array, cmap='Greys', interpolation='None')
                ax1.set_title('Original Image')
                im2 = ax2.imshow(image_array, cmap='Greys', interpolation='None', norm=Normalize(0,8))
                ax2.set_title('Normalized Image')
                plt.show()
                plt.close()


            except KeyboardInterrupt:
                break

    # Train neural net
    # Provide the name of the dataset file as arg, minus .csv extension
    def train(self, dataset):
        # Load the MNIST 100 training samples CSV file into a list
        training_data_list = self.read_csv(dataset)

        # Train the neural network on each trainingsample
        for record in training_data_list:
            # Split the record by the commas
            all_values = record.split(',')
            # Scale and shift the inputs from 0..255 to 0.01..1
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # Create the target output values (all 0.01, except the desired label which is 0.99)
            targets = np.zeros(self.o_nodes) + 0.01
            # All_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            # Train the network
            self.n.train(inputs, targets)
    
    # Test the neural net by querying it
    # Provide the name of the dataset file as arg, minus .csv extension
    def test(self, dataset):
        # Load the MNIST test samples CSV file into a list
        test_data_list = self.read_csv(dataset)

        # Scorecard list for how well the network performs, initially empty
        scorecard = []

        # Loop through all of the records in the test data set
        print(f'Results for the {dataset} dataset') if self.verbose else 0

        for record in test_data_list:
            # Split the record by the commas
            all_values = record.split(',')

            # The correct label is the first value
            actual_label = int(all_values[0])

            # Scale and shift the inputs
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # Query the network
            outputs = self.n.query(inputs)
            # The index of the highest value output corresponds to the label
            label = np.argmax(outputs)            

            # Append either a 1 or a 0 to the scorecard list
            correct = 0
            if (label == actual_label): # If nn got it right
                correct = 1
            else:
                correct = 0
            
            scorecard.append(correct) 

            print(f'Actual label was: {actual_label}, Network identified it as: {label}, {bool(correct)}') if self.verbose else 0

        # Calculate the performance score, the fraction of correct answers
        scorecard_array = np.asarray(scorecard)
        self.score = (scorecard_array.sum() / scorecard_array.size)*100
        print(f'Performance = {self.score}%') if self.verbose else 0

