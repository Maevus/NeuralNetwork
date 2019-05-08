import NeuralNetwork as nn
import json
from datetime import datetime
import random
import numpy as np
import csv

class NumberPredictior:
    def __init__(self, *args, **kwargs):
        numbers = []

        with open('letter-recognition.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)     
            for row in reader:
                inputs = list(map(int, row[1:len(row)]))
                numbers.append({
                    'inputs': inputs,
                    'target': [row[0]]
                })

        random.shuffle(numbers)

        self.trainDat = numbers[0:15999]
        self.testDat = numbers[16000:len(numbers)]

        self.neuralNetwork = nn.NeuralNetwork(16, 10, 26, 0.2)

        self.errorFile = open("Numbers_errors_{0}.txt".format(datetime.timestamp(datetime.now())), "w")
        self.predictFile = open("Numbers_predictions_{0}.txt".format(datetime.timestamp(datetime.now())), "w")

        self.train()
        self.test()
        self.tidy()

    def train(self):
        print("training...")
        for epoch in range(10):
            error = 0
            for d in self.trainDat:
                error = self.neuralNetwork.train(d['inputs'], d['target'])
                if (random.randint(0,1) < 0.10):
                    self.neuralNetwork.updateWeightsAndBiases()
                
                self.errorFile.write("GD Iteration: {0}, Error: {1} \n".format(epoch, np.array2string(error.flatten())))
    
    def test(self):
        print("testing...")
        for t in self.testDat:
            prediction = self.neuralNetwork.predict(t['inputs'])
            target = t['target']
            error = np.subtract(target, prediction)
            self.predictFile.write("Predicted: {0}  |   Expected:{1}   |   Error:{2}\n".format(prediction, target, error))
    
    def tidy(self): 
        print("done.")
        self.errorFile.close
        self.predictFile.close

if __name__=='__main__':
    NumberPredictior()