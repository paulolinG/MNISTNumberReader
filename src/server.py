from flask import Flask, request, jsonify
from flask_caching import Cache
from flask_cors import CORS
from .arch.neural_network import NeuralNetwork
import numpy as np
import gc

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

layer_config = [2, 2, 1]  
input_size = 728  
activations = ["relu", "relu", "softmax"]
learning_rate = 0.01
neural_net = NeuralNetwork(layer_config, input_size, activations)

@app.route('/api/set_parameters', methods=['POST'])
def set_parameters():
    global layer_config
    global learning_rate
    global activations
    global neural_net
    data = request.json
    if ('learning_rate' not in data) or ('layer_config' not in data):
        return jsonify({'error': 'No input data provided'}), 400
    try:
        learning_rate = data['learning_rate']
        layer_config = data['layer_config'] / 1000
        del neural_net
        gc.collect()

        activations = ["relu" for i in range(len(layer_config) - 1)]
        activations.append("softmax")

        neural_net = NeuralNetwork(layer_config, input_size, activations)
        return jsonify('successfully reconfigured neural network')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/architecture', methods=['GET'])
@cache.cached(timeout=60)  # Cache for 60 seconds
def get_architecture():
    architecture = neural_net.get_architecture()
    return jsonify(architecture)


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    if 'input' not in data:
        return jsonify({'error': 'No input data provided'}), 400
    try:
        input_data = np.array(data['input']).reshape(-1,1)
        predictions = neural_net.predict(input_data)
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train():
    data = request.json
    if 'input' not in data or 'labels' not in data:
        return jsonify({'error': 'Insufficient training data.'}), 400
    try:
        input_data = np.array(data['input']).reshape(-1, 1)  # Reshape as column vector
        labels = np.array(data['labels']).reshape(-1, 1)    # Reshape as column vector
        neural_net.train(input_data, labels, learning_rate)
        cost = neural_net.computeCost(labels)
        return jsonify({'status': 'Training step completed', 'cost': cost})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/api/status', methods=['GET'])
def status():
    status = neural_net.get_status()
    return jsonify(status)


if __name__ == '__main__':
    app.run(debug=True)

