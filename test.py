import csv
import random

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

train = numbers[0:15999]
test = numbers[16000:len(numbers)]

print (test[0]['inputs'])