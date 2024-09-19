import numpy as np
from .weight_matrix import WeightMatrix

class NeuronLayer:
    """
    a neuron layer can be represented by the formula activation( W * x + b ), where
    W is the weight matrix for the following layer, x is either the inputs or the activation
    values, and b is the bias vector
    """ 

    """
        neuronsPrev is the number of neurons in the previous layer
        neurons is the number of neurons in the current layer
        biases is a vector of the biases used for this layer
    """

    def __init__(self, neuronsPrev, neurons, activation):
        self.biases = np.zeros(shape=(neurons,1))
        self.weightMatrix = WeightMatrix(neuronsPrev, neurons)
        self.errorVector = None
        self.activationValues = None
        self.weightCosts = None
        self.activation = activation

    """
    Computes the corresponding errorVector for this layer 
        weightMatrix is the weight matrix for the next layer
        errorVector is the errors of the neurons for the next layer
    """

    def computeErrorVector(self, weightMatrix, errorVector):
        reluDerivative = (self.activationValues > 0).astype(float)
        self.errorVector = np.matmul(np.transpose(weightMatrix.weights), errorVector) * reluDerivative

    """
    Compute the partial derivative of the cost w.r.t the weights in the weight matrix
        activationValues is the activationValues of the previous layer
    """

    def computeCostWeightDerivative(self, activationValues):
        self.weightCosts = np.matmul(self.errorVector, np.transpose(activationValues))

        # verify the deimneions of the weight costs
        assert len(self.weightCosts) == len(self.weightMatrix.weights)
        assert len(self.weightCosts[0]) == len(self.weightMatrix.weights[0])

    """
    Computes the activation values for this given layer
        preActivationValues is the activation values of the previous layer
    """

    def computeActivationValues(self, prevActivationValues):
        z = np.matmul(self.weightMatrix.weights, prevActivationValues) + self.biases
        if self.activation == 'relu':
            self.activationValues = np.maximum(0, z)
        elif self.activation == 'softmax':
            exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
            self.activationValues = exp_z / np.sum(exp_z, axis=0, keepdims=True)
        else:
            raise ValueError("Unsupported activation function")

    """
    Computes the errorVector for this layer 
        costGradient is the change in cost over the change in all of the activation values
        in the last layer

    Should only be called if the layer is the output layer in the neural network
    """

    def computeErrorVectorLastLayer(self, labels):
        self.errorVector = self.activationValues - labels
    

