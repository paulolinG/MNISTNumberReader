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
                self.neuralNetwork[i] = NeuronLayer(inputSize, NNConfig[0], activations[0], "Input")
            elif i == len(NNConfig) - 1:
                self.neuralNetwork[i] = NeuronLayer(NNConfig[i-1], NNConfig[i], activations[i], "Output")
            else:
                self.neuralNetwork[i] = NeuronLayer(NNConfig[i-1], NNConfig[i], activations[i], "Dense")
        self.is_training = False
        self.current_labels = None

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
    Computes the cost of the output
        labels is the expected output of the corresponding inputs
    """

    def computeCost(self, labels):
        predictions = self.neuralNetwork[-1].activationValues
        epsilon = 1e-12
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        ce_loss = -np.mean(np.sum(labels * np.log(predictions), axis=0))
        return ce_loss

    
    """
    Performs backpropagation on the array, calculating the error contributions of all the nodes in the neural network
        labels is the target values to use for backpropagation
    """

    def backProp(self, labels):
        # compute the errors of the last layer
        lastLayer = self.neuralNetwork[-1]
        lastLayer.computeErrorVectorLastLayer(labels)

        # compute the errors of the reamining layers
        for i in range(2, len(self.neuralNetwork) + 1):
            layerAfter = self.neuralNetwork[-i + 1]
            layer = self.neuralNetwork[-i]
            layer.computeErrorVector(layerAfter.weightMatrix, layerAfter.errorVector)


    """
    Computes dC / dW for each weight in the neural network
        inputs is the inputs used to compute dC / dW for the weights in the first layer
    """


    def computeWeightErrors(self, inputs):
        # calculate the cost derivatives with resepct to the weights
        for i in range(len(self.neuralNetwork)):
            layer = self.neuralNetwork[i]
            activationValues = inputs if i == 0 else self.neuralNetwork[i-1].activationValues
            layer.computeCostWeightDerivative(activationValues)


    """
    Updates the weight and biases based on their error contributions
        learningRate is the learning rate used to update the weights and biases
    """


    def update_weights_and_biases(self, learningRate):
        for layer in self.neuralNetwork:
            layer.weightMatrix.weights = layer.weightMatrix.weights - (learningRate * layer.weightCosts)
            bias_grad = np.sum(layer.errorVector, axis = 1, keepdims=True)
            layer.biases = layer.biases - (learningRate * bias_grad)


    def get_architecture(self):
        architecture = {
            "layers": []
        }
        for index, layer in enumerate(self.neuralNetwork):
            layer_info = {
                "layer_number": index + 1,
                **layer.serialize()
            }
            architecture["layers"].append(layer_info)
        return architecture

    def get_status(self):
        if self.is_training:
            status = {
                "message": "Training in progress...",
                "cost": None
            }
        else:
            cost = self.computeCost(self.current_labels) if self.current_labels is not None else None
            status = {
                "message": "Model is ready.",
                "cost": cost
            }
        return status
    
    
    def predict(self, input_data):
        self.forwardPass(input_data)
        return self.neuralNetwork[-1].activationValues