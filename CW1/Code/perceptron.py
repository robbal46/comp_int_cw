# Import the numpy library for matrix math
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, name, weights, bias):
        self.name = name
        self.weights = weights
        self.bias = bias

    # Calculates the output of the perceptron
    def output(self, inputs):
        inputs_arr = np.array(inputs)
        weights_arr = np.array(self.weights)
        summed = np.dot(inputs_arr, weights_arr)
        summed = summed + self.bias
        output = 1 if summed > 0 else 0
        return output

    # Returns a list of tuples to plot linear separator
    def separator(self):
        func = lambda x: -(self.weights[0] / self.weights[1])*x - (self.bias / self.weights[1])
        return [(0, func(0)), (1, func(1))]

    def state_space(self, input_list):

        plt.figure()

        for input in input_list:        
            output = self.output(input)

            colours = ['red', 'green']
            plt.scatter(input[0], input[1], s=50, color=colours[output], zorder=3)
            plt.xlim(-1, 2)
            plt.ylim(-1, 2)
            plt.xlabel('Input 1')
            plt.ylabel('Input 2')
            plt.title(f'State Space of Input Vector - {self.name}')
            plt.grid(True, linewidth=1, linestyle=':')
            plt.tight_layout()

        separator = self.separator()
        plt.axline(separator[0], separator[1], label=self.name)

        plt.legend()
        plt.show()



# Implements a 2 layer perceptron network
class MLP:

    def __init__(self, name, hidden_layer, output_layer):
        self.name = name
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
    
    # Calculates the output from the output perceptron 
    def output(self, inputs):
        # Call output method on each perceptron object in the list
        hidden_outputs = list(map(lambda x: x.output(inputs), self.hidden_layer))

        output = self.output_layer.output(hidden_outputs)
        return output

    # Plots the state space of the input vector
    def state_space(self, input_list):

        plt.figure()

        for input in input_list:
            output = self.output(input)

            colours = ['red', 'green']
            plt.scatter(input[0], input[1], s=50, color=colours[output], zorder=3)
            plt.xlim(-1, 2)
            plt.ylim(-1, 2)
            plt.xlabel('Input 1')
            plt.ylabel('Input 2')
            plt.title(f'State Space of Input Vector - {self.name}')
            plt.grid(True, linewidth=1, linestyle=':')
            plt.tight_layout()

        for h in self.hidden_layer:
            sep = h.separator()
            plt.axline(sep[0], sep[1], label=h.name)

        plt.legend()
        plt.show()


# Main
if __name__ == '__main__':

    # All possible inputs (x1, x2)
    all_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]


    #p = Perceptron('AND', [1.0, 1.0], -1.5)
    #p = Perceptron('NAND', [-1.0, -1.0], 1.5)
    p = Perceptron('OR', [1.0, 1.0], -0.5)
    p.state_space(all_inputs)

    # h1, h2 = Perceptron('Hidden 1 (OR)', [1.0, 1.0], -0.5), Perceptron('Hidden 2 (NAND)', [-1.0, -1.0], 1.5)
    # o = Perceptron('Output', [1.0, 1.0], -1.5)
    # mlp = MLP('XOR', [h1, h2], o)
    # mlp.state_space(all_inputs)
