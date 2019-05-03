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
        print("**************************************************\n\n")

    def applyActivationFunc(self, matrix):
        try:
            return map(self.sigmoid, matrix)
        except:
            print("Error mapping activation function to weight matrix.")    


    def feedForward(self, inputs):
        # Calc hidden layer
        hidden = np.multiply(self.weightsIH, inputs)
        np.add(hidden, self.biasHidden)
        print("hidden layer weight matrix {0}".format(hidden))
        self.applyActivationFunc(hidden)      

        # Calc output layer
        output = np.multiply(self.weightsHO, hidden)
        np.add(output, self.biasOutput)
        self.applyActivationFunc(output)

        print("output layer: {0}".format(output))
        print("shape: {0}".format(output.shape))

        return output

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def sigmoidDerivative(self, sx):
        return sx*(1.0 - sx)




# Setup
neuralNetwork = NeuralNetwork(2, 2, 1, 0.1)
inputs = [0,1]

# Run 
neuralNetwork.feedForward(inputs)
