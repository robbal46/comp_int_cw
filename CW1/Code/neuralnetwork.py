# Import scipy.special for the sigmoid function expit()
import numpy as np
import scipy.special

# Neural network class definition
class NeuralNetwork:
    # Init the network, this gets run whenever we make a new instance of this class
    def __init__ (self, structure, learning_rate):
        # Set the number of nodes in each input, hidden and output layer
        # self.i_nodes = input_nodes
        # self.h_nodes = hidden_nodes
        # self.o_nodes = output_nodes

        self.structure = structure   
        self.layers = len(self.structure)     

        # Weight matrices, wih (input -> hidden) and who (hidden -> output)
        self.wih = np.random.normal(0.0, pow(self.structure[1], -0.5), (self.structure[1], self.structure[0]))
        self.who = np.random.normal(0.0, pow(self.structure[2], -0.5), (self.structure[2], self.structure[1]))
        
        # Set the learning rate
        self.lr = learning_rate
        # Set the activation function, the logistic sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)

    def update_lr(self, lr):
        self.lr = lr

    # Train the network using back-propagation of errors
    def train(self, inputs_list, targets_list):
        # Convert inputs into 2D arrays
        inputs_array = np.array(inputs_list, ndmin=2).T
        targets_array = np.array(targets_list, ndmin=2).T 
        # Calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs_array)
        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # Calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        # Current error is (target - actual)
        output_errors = targets_array - final_outputs
        # Hidden layer errors are the output errors, split by the weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
        np.transpose(hidden_outputs))
        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
        np.transpose(inputs_array))

    # Query the network
    def query(self, inputs_list):
        # Convert the inputs list into a 2D array
        inputs_array = np.array(inputs_list, ndmin=2).T
        # Calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs_array)
        # Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # Calculate signals into final layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # Calculate outputs from the final layer
        final_outputs =self.activation_function(final_inputs)
        return final_outputs





