import numpy as np
import pandas as pd
import _pickle

def deepcopy(x):
	return _pickle.loads(_pickle.dumps(x, -1))

def grab(path):
	return pd.read_csv(path + 'train.csv'), pd.read_csv(path + 'test.csv')

def initalize(dims):
	weights, bias = [], np.random.rand((len(dims)))
	for layer in dims:
		weights.append(np.random.rand(layer[0],layer[1]))
	return np.array(weights, dtype=object), np.array(bias)

def mutate(weights, bias, mutationChance, mutationAmount):
	for layer in range(len(bias)):
		bias[layer][row] += np.random.uniform(-mutationAmount, mutationAmount)
		for row in range(len(weights[layer])):
			if np.random.uniform(0, 1) < mutationChance:
				weights[layer][row] += np.random.uniform(-mutationAmount, mutationAmount)
	return weights, bias

def relu(x):
	return np.max(0, x)

def cost(netOutputs, realOutputs):
	return (realOutputs - netOutputs)**2

def layer(inputs, weights, bias):
	return relu(np.dot(inputs, weights) + bias)

def network(inputs, weights, bias):
	for layer in range(len(inputs)):
		inputs = layer(inputs, weights[layer], bias[layer])
	return inputs

def evolve(inputs, weights, bias, outputs, batchSize, mutationChance=0.1, mutationAmount=0.5):
	minError = network(inputs, weights, bias)
	bestWeights, bestBias = weights, bias
	for network in range(batchSize):
		weightsM, biasM = mutate(deepcopy(weights), deepcopy(bias), mutationChance, mutationAmount)
		netOutputs = network(inputs, weightsM, biasM)
		error = cost(netOutputs, outputs)

		if error < minError:
			bestWeights, bestBias = weightsM, biasM
			minError = error
	
	return bestWeights, bestBias
