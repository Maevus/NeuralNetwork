'''
April 2019
A wee neural network.
Maeve Lynskey - 07257724
'''

import numpy as np

class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes 
        self.outputNodes = outputNodes

        # init weights matrices with random weights between 0 and 1.
        self.weightsIH = np.random.random((self.hiddenNodes, self.inputNodes))
        self.weightsHO = np.random.random((self.outputNodes, self.hiddenNodes))

        # init biases with random values between 0 and 1.
        self.biasHidden = np.random.random((self.hiddenNodes, 1))
        self.biasOutput = np.random.random((self.outputNodes, 1))

        self.learningRate = learningRate

        print("Activating NN...\ninput nodes: {0}\nhidden nodes: {1}\noutput nodes: {2}\nlearning rate: {3}"
            .format(self.inputNodes, self.hiddenNodes, self.outputNodes, self.learningRate))
        print("map(sigmoid, matrix)**************************************************\n\n")
 
    def feedForward(self, inputs):
        # Calc hidden layer
        inputs = np.vstack(inputs)
        hidden = np.dot(self.weightsIH, inputs)
        hidden = np.add(hidden, self.biasHidden)
        hidden = self.sigmoid(hidden)     
        print("squashed hidden layer:\n{0}".format(hidden))

        # Calc output layer
        output = np.dot(self.weightsHO, hidden)
        output = np.add(output, self.biasOutput)
        output = self.sigmoid(output) 
        print("squashed output:\n{0}".format(output))
        print("shape: {0}".format(output.shape))

        return output

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def sigmoidDerivative(self, sx):
        return sx*(1.0 - sx)


# Setup
neuralNetwork = NeuralNetwork(2, 2, 1, 0.1)
inputs = np.array([0,1])

# Run 
neuralNetwork.feedForward(inputs)
