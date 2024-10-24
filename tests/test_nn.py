from src.arch.neural_network import NeuralNetwork

def test_basic_nn():
    nn = NeuralNetwork([1], 10)
    nn.forwardPass([[1,1,1,1,1,1,1,1,1,1]])
    print("test nn 1")
    for layer in nn.neuralNetwork:
        print("\n")
        print(layer.activationValues)
    print("\n")

def test_basic_nn2():
    nn = NeuralNetwork([2], 10)
    nn.forwardPass([[1,1,1,1,1,1,1,1,1,1]])
    print("test nn 2")
    for layer in nn.neuralNetwork:
        print("\n")
        print(layer.activationValues)
    print("\n")

def test_basic_nn3():
    nn = NeuralNetwork([5,5], 10)
    nn.forwardPass([[1,1,1,1,1,1,1,1,1,1]])
    print("test nn 3")
    for layer in nn.neuralNetwork:
        print("\n")
        print(layer.activationValues)
    print("\n")

def test_compute_last_layer():
    nn = NeuralNetwork([7,5], 10)
    nn.forwardPass([[1,1,1,1,1,1,1,1,1,1]])
    vector = nn.computCostGradientLastLayer([[1,1,1,1,1]])
    print("test compute cost gradient last layer")
    print(vector)
    print("\n")
    
def test_back_propagate():
    nn = NeuralNetwork([4,4], 10)
    nn.forwardPass([[1,1,1,1,1,1,1,1,1,1]])
    nn.backProp(labels=[[1,1,1,1]])
    print("test back prop")
    for layer in nn.neuralNetwork:
        print("\n")
        print(layer.errorVector)
    print("\n")

def test_weight_errors():
    nn = NeuralNetwork([4,4], 10)
    nn.forwardPass([[1,1,1,1,1,1,1,1,1,1]])
    nn.backProp(labels=[[1,1,1,1]])
    nn.computeWeightErrors(inputs=[[1,1,1,1,1,1,1,1,1,1]])
    print("test weight errors")
    for layer in nn.neuralNetwork:
        print("\n")
        print(layer.weightCosts)
    print("\n")