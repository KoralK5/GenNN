import numpy as np
import neuralNetwork as nn
import pandas as pd

path = '.\\GenNN\\'
dataPath = path + 'Data\\'
modelPath = path + 'Model\\'

dims = [(784, 18), (392, 18), (10, 18)]

train, test = nn.grab(dataPath)
weights, bias = nn.initalize(dims)

batchSize = 25
mutationChance = 0.1
mutationAmount = 0.5

allInputs = df.iloc[:, 1:].values
allOutputs = df.iloc[:, 0].values

for img in range(len(allOutputs)):
	inputs = allInputs[img]
	outputs = allOutputs[img]
	params = (inputs, weights, bias, outputs, batchSize, mutationChance, mutationAmount)

	weights, bias = evolve(*params)

	np.save(modelPath+'weights.npy', weights)
	np.save(modelPath+'bias.npy', bias)
