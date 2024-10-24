from src.arch import weight_matrix as wm

def testWm1():
    weights = wm.WeightMatrix(10, 5)
    assert len(weights.weights) == 5
    assert len(weights.weights[0]) == 10
    print(weights.weights)