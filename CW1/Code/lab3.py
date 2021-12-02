from neuralnetwork import NeuralNetwork

n = NeuralNetwork(3,20,1,0.3)

training_iterations = 1000

#inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
inputs = [[0.0, 0.0, 0.0], 
        [0.0, 0.0, 1.0], 
        [0.0, 1.0, 0.0], 
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0], 
        [1.0, 1.0, 0.0], 
        [1.0, 1.0, 1.0]]

#targets = [1, 1, 1, 0]
targets = [0,1,1,0,1,0,0,1] #XOR

for iters in range(0, training_iterations):
    for i,j in zip(inputs, targets):
        n.train(i, j)

for k in inputs:
    print(n.query(k))