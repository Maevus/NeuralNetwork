import NeuralNetwork as nn
import json
from datetime import datetime
import random
import numpy as np

class VectorPredictior:
    def __init__(self, *args, **kwargs):
        with open("vectors.json", "r") as f:
             raw_data = f.read()
        
        data = json.loads(raw_data)

        self.neuralNetwork = nn.NeuralNetwork(4, 5, 1, 0.2)

        self.errorFile = open("Vectors_errors_{0}.txt".format(datetime.timestamp(datetime.now())), "w")
        self.predictFile = open("Vectors_predictions_{0}.txt".format(datetime.timestamp(datetime.now())), "w")
        self.trainDat = data[0:150]
        self.testDat = data[50:100]

        self.train()
        self.test()
        self.tidy()

    def train(self):
        print("training...")
        for epoch in range(1000):
            error = 0
            for d in self.trainDat:
                error = self.neuralNetwork.train(d['inputs'], d['outputs'])
                if (random.randint(0,1) < 0.10):
                    self.neuralNetwork.updateWeightsAndBiases()
                
                self.errorFile.write("GD Iteration: {0}, Error: {1} \n".format(epoch, np.array2string(error.flatten())))
    
    def test(self):
        print("testing...")
        for t in self.testDat:
            prediction = self.neuralNetwork.predict(t['inputs'])
            target = t['outputs']
            error = np.subtract(target, prediction)
            self.predictFile.write("Predicted: {0}  |   Expected:{1}   |   Error:{2}\n".format(prediction, target, error))
    
    def tidy(self): 
        print("done.")
        self.errorFile.close
        self.predictFile.close

if __name__=='__main__':
    VectorPredictior()