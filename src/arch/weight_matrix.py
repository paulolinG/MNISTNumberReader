import numpy as np

class WeightMatrix:
    """
    This class represents a weight matrix for neuron layer
    """

    """
        numNueronsPrev is the number of neurons in the previous layer
        numNeuronsCurr is the number of nuerons in this layer
    """
    def __init__(self, numNeuronsPrev, numNeurons):
        self.sizeIn = numNeuronsPrev
        self.sizeOut = numNeurons
        self.weights = self.he_normal_init()


    def he_normal_init(self):
        stddev = np.sqrt(2.0 / self.sizeIn)
        return np.random.normal(0, stddev, (self.sizeOut, self.sizeIn))


    def basicTest(self):
        return np.ones((self.sizeOut, self.sizeIn))
    

    def serialize(self):
        return {
            "weights": self.weights.tolist()
        }