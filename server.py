from flask import Flask, request, jsonify, render_template
from flask_caching import Cache
from flask_cors import CORS
from src.arch.neural_network import NeuralNetwork
import numpy as np
import gc
import tensorflow as tf
import tensorflow_datasets as tfds

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

layer_config = [2, 2, 1]  
input_size = 728  
activations = ["relu", "relu", "softmax"]
learning_rate = 0.01
neural_net = NeuralNetwork(layer_config, input_size, activations)

@app.route('/', methods=['GET'])
def render_page():
    return render_template('index.html')

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
        neural_net.is_training = True
        
        ds = tfds.load('mnist', split='train', shuffle_files=True)
        learningRate = 0.01
        epochs = 10

        ds = ds.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
        network = NeuralNetwork(NNConfig=[16, 16, 10], inputSize=784, activations=['relu', 'relu', 'softmax'])

        # Training loop:
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            for batch in ds:
                images, labels = batch["image"].numpy(), batch["label"].numpy()
                
                images = images.reshape(images.shape[0], -1).T
                images = images / 255.0
                labels_one_hot = np.eye(10)[labels].T

                network.forwardPass(inputs=images)

                loss = network.computeCost(labels=labels_one_hot)
                epoch_loss += loss
                num_batches += 1

                network.backProp(labels=labels_one_hot)

                network.computeWeightErrors(inputs=images)

                network.update_weights_and_biases(learningRate=learningRate)
                
            test_ds = tfds.load('mnist', split='test', shuffle_files=False)
            test_batch_size = 1000 
            test_ds = test_ds.batch(test_batch_size).prefetch(tf.data.AUTOTUNE)

            all_predictions = []
            all_labels = []

            for batch in test_ds:
                test_images, test_labels = batch["image"].numpy(), batch["label"].numpy()

                test_images = test_images.reshape(test_images.shape[0], -1).T  
                test_images = test_images / 255.0  

                network.forwardPass(inputs=test_images)

                predictions = network.neuralNetwork[-1].activationValues

                all_predictions.append(predictions)
                all_labels.append(test_labels)

            all_predictions = np.concatenate(all_predictions, axis=1)
            all_labels = np.concatenate(all_labels, axis=0)

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

