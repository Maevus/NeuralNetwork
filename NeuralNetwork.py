'''
April 2019
A wee neural network.
Maeve Lynskey - 07257724
'''

import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoidDerivative(sx):
    return sx*(1.0 - sx)

class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes 
        self.outputNodes = outputNodes

        # init random weights matrices
        self.weightsIH = np.random.random((self.hiddenNodes, self.inputNodes))
        self.weightsHO = np.random.random((self.outputNodes, self.hiddenNodes))

        self.biasHidden = np.random.random((self.hiddenNodes, 1))
        self.biasOutput = np.random.random((self.outputNodes, 1))

        self.learningRate = learningRate
    
    def feedForward(input):
        return output
    