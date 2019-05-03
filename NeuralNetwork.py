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
 

    # TODO change name to predict or something.  
    def feedforward(self, vstackedinputs):
        # Calc hidden layer
        hidden = np.dot(self.weightsIH, vstackedinputs)
        hidden = self.sigmoid(np.add(hidden, self.biasHidden))
        print("squashed hidden layer:\n{0}".format(hidden))

        # Calc output layer
        output = np.dot(self.weightsHO, hidden)
        output = self.sigmoid(np.add(output, self.biasOutput))
        print("squashed output:\n {0}".format(output))
        print("shape: {0}".format(output.shape))

        return output



    def train(self, inputs, targets):
        #output = self.feedforward(np.vstack(inputs))

        # Calc hidden layer
        hidden = np.dot(self.weightsIH, np.vstack(inputs))
        hidden = self.sigmoid(np.add(hidden, self.biasHidden))
        print("squashed hidden layer:\n{0}".format(hidden))

        # Calc output layer
        output = np.dot(self.weightsHO, hidden)
        output = self.sigmoid(np.add(output, self.biasOutput))
        print("squashed output:\n {0}".format(output))

        #### Calculate output layer errors ####
        outputErrors = np.subtract(np.vstack(targets), output)

        # Calculate Gradient
        gradient = self.sigmoidDerivative(output)
        # Muliply output deltas with output errors
        gradient = np.multiply(outputErrors, gradient)
        # Learning rate
        gradient = np.multiply(self.learningRate, gradient)
        print("gradient: {0}".format(gradient))

        # Calculate weight deltas
        hiddenTransposed = np.transpose(hidden)
        weightsHOdeltas = np.dot(gradient, hiddenTransposed)
        # Update weights and bias
        self.weightsHO = np.add(self.weightsHO, weightsHOdeltas)
        self.biasOutput = np.add(self.biasOutput, gradient)

        #### Calculate hidden layer errors ####

        # Backpropagation step.
        weightsHOTransposed = np.transpose(self.weightsHO)
        hiddenErrors = np.dot(weightsHOTransposed, outputErrors)

        # Hidden gradient
        hiddenGradient = self.sigmoidDerivative(hidden)
        hiddenGradient = np.multiply(hiddenErrors, hiddenGradient)
        hiddenGradient = np.multiply(self.learningRate, hiddenGradient)
        
        # Hidden deltas
        inputsTransposed = np.transpose(inputs)
        weightsIHdeltas = np.dot(inputsTransposed, hiddenGradient)

        # Update weights and bias
        self.weightsIH = np.add(self.weightsIH, weightsIHdeltas)
        self.biasHidden = np.add(self.biasHidden, hiddenGradient)


    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))


    def sigmoidDerivative(self, sx):
        return sx*(1.0 - sx)

# Setup
neuralNetwork = NeuralNetwork(2, 2, 2, 0.1)
inputs = np.array([0,1])
targets = [1, 2]

# Run 
neuralNetwork.train(inputs, targets)
