import numpy as np
from .neuron_layer import NeuronLayer

class NeuralNetwork:
    """
    This class represents a nerual network
    """

    """
        numLayers is the number of layers in the neural network
        NNConfig is an array of integers where the NNConfig[i] is the number of neurons in the i'th layer
        inputSize is the number of inputs fed into the neural network
    """
    def __init__(self, NNConfig, inputSize, activations):
        self.neuralNetwork = np.empty(len(NNConfig), dtype=object)
        for i in range(len(NNConfig)):
            if i == 0:
                self.neuralNetwork[i] = NeuronLayer(inputSize, NNConfig[0], activation=activations[0])
            else:
                self.neuralNetwork[i] = NeuronLayer(NNConfig[i-1], NNConfig[i], activation=activations[i])

    """
    Forward propagation to compute the activation values
        inputs fed in as one row vector of inputs
    """
            
    def forwardPass(self, inputs):
        for i in range(len(self.neuralNetwork)):
            layer = self.neuralNetwork[i]
            if i == 0:
                layer.computeActivationValues(inputs)
            else:
                layer.computeActivationValues(self.neuralNetwork[i-1].activationValues)

    """
    Computes the cost of the output with mse
        labels is the expected output of the corresponding inputs
    """

    def computeCost(self, labels):
        predictions = self.neuralNetwork[-1].activationValues
        epsilon = 1e-12
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        ce_loss = -np.mean(np.sum(labels * np.log(predictions), axis=0))
        return ce_loss

    
    """
    Performs backpropagation on the array, calculating the error contributions of all the nodes in the nn
        inputs: backProp(self, labels)
    """

    def backProp(self, labels):
        # compute the errors of the last layer
        batches = labels.shape[1]
        numClasses = labels.shape[0]
        lastLayer = self.neuralNetwork[-1]
        lastLayer.computeErrorVectorLastLayer(labels)

        # compute the errors of the reamining layers
        for i in range(2, len(self.neuralNetwork) + 1):
            layerAfter = self.neuralNetwork[-i + 1]
            layer = self.neuralNetwork[-i]
            layer.computeErrorVector(layerAfter.weightMatrix, layerAfter.errorVector)


    def computeWeightErrors(self, inputs):
        # calculate the cost derivatives with resepct to the weights
        for i in range(len(self.neuralNetwork)):
            layer = self.neuralNetwork[i]
            activationValues = inputs if i == 0 else self.neuralNetwork[i-1].activationValues
            layer.computeCostWeightDerivative(activationValues)


    def update_weights_and_biases(self, learningRate):
        for layer in self.neuralNetwork:
            layer.weightMatrix.weights = layer.weightMatrix.weights - (learningRate * layer.weightCosts)
            bias_grad = np.sum(layer.errorVector, axis = 1, keepdims=True)
            layer.biases = layer.biases - (learningRate * bias_grad)
    
            
