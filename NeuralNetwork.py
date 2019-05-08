'''
April 2019
A wee neural network.
Maeve Lynskey - 07257724
'''

import numpy as np
import json
import random
from datetime import datetime

class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes 
        self.outputNodes = outputNodes

        # init weights matrices with random weights between 0 and 1.
        self.weightsIH = np.random.random((self.hiddenNodes, self.inputNodes))
        self.weightsHO = np.random.random((self.outputNodes, self.hiddenNodes))

        # init zeroed delta matrices
        self.weightsIHdeltas = np.zeros((self.hiddenNodes, self.inputNodes))
        self.weightsHOdeltas = np.zeros((self.outputNodes, self.hiddenNodes))

        # store gradient
        self.gradient = np.zeros((self.outputNodes, 1))
        self.hiddenGradient = np.zeros((self.hiddenNodes, 1))
        
        # init biases.
        self.biasOutput = np.random.random((self.outputNodes, 1))
        self.biasHidden = np.random.random((self.hiddenNodes, 1))

        self.learningRate = learningRate

        print("Activating NN...\ninput nodes: {0}\nhidden nodes: {1}\noutput nodes: {2}\nlearning rate: {3}"
            .format(self.inputNodes, self.hiddenNodes, self.outputNodes, self.learningRate))
        print("**************************************************\n\n")
 

    def predict(self, inputs):
        # Calc hidden layer
        hidden = np.dot(self.weightsIH, np.vstack(inputs))
        hidden = self.sigmoid(np.add(hidden, self.biasHidden))

        # Calc output layer
        output = np.dot(self.weightsHO, hidden)
        output = self.sigmoid(np.add(output, self.biasOutput))

        return output


    def train(self, inputs, targets):
        # Calc hidden layer
        h = np.dot(self.weightsIH, np.vstack(inputs))
        hidden = self.sigmoid(np.add(h, self.biasHidden))

        # Calc output layer
        o = np.dot(self.weightsHO, hidden)
        output = self.sigmoid(np.add(o, self.biasOutput))

        #### Calculate output layer errors ####
        outputErrors = np.subtract(np.vstack(targets), output)

        # Calculate Gradient
        outputDerivative = self.sigmoidDerivative(output)
        # Muliply output deltas with output errors & learning rate
        self.gradient = np.multiply(self.learningRate, np.multiply(outputErrors, outputDerivative))
        
        # Calculate weight deltas
        self.weightsHOdeltas = np.multiply(self.gradient, np.transpose(hidden)) 

        #### Calculate hidden layer errors ####

        # Backpropagation step.
        weightsHOTransposed = np.transpose(self.weightsHO)
        hiddenErrors = np.dot(weightsHOTransposed, outputErrors)

        # Hidden gradient
        hg = np.multiply(hiddenErrors, self.sigmoidDerivative(hidden))
        self.hiddenGradient = np.multiply(self.learningRate, hg)
        
        # Hidden deltas
        self.weightsIHdeltas = np.multiply(self.hiddenGradient, inputs) 

        return outputErrors


    def updateWeightsAndBiases(self):
            self.weightsHO = np.add(self.weightsHO, self.weightsHOdeltas)
            self.weightsIH = np.add(self.weightsIH, self.weightsIHdeltas)
            
            self.biasOutput = np.add(self.biasOutput, self.gradient)
            self.biasHidden = np.add(self.biasHidden, self.hiddenGradient)


    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def sigmoidDerivative(self, sx):
        return sx*(1.0 - sx)


if __name__ == "__main__":
    
    with open("xor.json", "r") as f:
        raw_data = f.read()
    data = json.loads(raw_data)

    neuralNetwork = NeuralNetwork(2, 2, 1, 0.2)

    
    file = open("errors_{0}.txt".format(datetime.timestamp(datetime.now())), "w")

    # Train
    # Gradient Descent
    for epoch in range(20000):
        error = 0
        for d in data:
            error = neuralNetwork.train(d['inputs'], d['targets'])
            if (random.randint(0,1) < 0.10):
                neuralNetwork.updateWeightsAndBiases()
        file.write("GD Iteration: {0}, Error: {1} \n".format(epoch, np.array2string(error.flatten())))
        

    # Tidy
    file.close

    # Test 
    print(neuralNetwork.predict([0,0]))
    print(neuralNetwork.predict([1,1]))
    print(neuralNetwork.predict([1,0]))
    print(neuralNetwork.predict([0,1]))