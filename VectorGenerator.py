import random
import math
import json

vectors = []

for i in range(200):
    inputs = [random.uniform(-1,1) for _ in range(4)]
    vectors.append({
    'inputs': inputs,
    'outputs': [math.sin(inputs[0]-inputs[1]+inputs[2]-inputs[3])]
    })

jsons = json.dumps(vectors)

with open("vectors.json", "w") as f:
    f.write(jsons)