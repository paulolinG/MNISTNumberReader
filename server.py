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

layer_config = [2, 2, 10]  
input_size = 784  
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
        learning_rate = int(data['learning_rate']) / 1000
        layer_config = [int(data['layer_config'][0]) for i in range(len(data['layer_config']))]
        layer_config.append(10)
        del neural_net
        gc.collect()

        activations = ["relu" for i in range(len(layer_config) - 1)]
        activations.append("softmax")

        neural_net = NeuralNetwork(layer_config, input_size, activations)
        return jsonify('successfully reconfigured neural network')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/architecture', methods=['GET'])
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
        print(e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train():
    try:
        neural_net.is_training = True
        
        ds = tfds.load('mnist', split='train', shuffle_files=True)
        epochs = 10

        ds = ds.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)

        # Training loop:
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            for batch in ds:
                images, labels = batch["image"].numpy(), batch["label"].numpy()
                
                images = images.reshape(images.shape[0], -1).T
                images = images / 255.0
                labels_one_hot = np.eye(10)[labels].T

                neural_net.forwardPass(inputs=images)

                loss = neural_net.computeCost(labels=labels_one_hot)
                epoch_loss += loss
                num_batches += 1

                neural_net.backProp(labels=labels_one_hot)

                neural_net.computeWeightErrors(inputs=images)

                neural_net.update_weights_and_biases(learning_rate)
                
            test_ds = tfds.load('mnist', split='test', shuffle_files=False)
            test_batch_size = 1000 
            test_ds = test_ds.batch(test_batch_size).prefetch(tf.data.AUTOTUNE)

            all_predictions = []
            all_labels = []

            for batch in test_ds:
                test_images, test_labels = batch["image"].numpy(), batch["label"].numpy()

                test_images = test_images.reshape(test_images.shape[0], -1).T  
                test_images = test_images / 255.0  

                neural_net.forwardPass(inputs=test_images)

                predictions = neural_net.neuralNetwork[-1].activationValues

                all_predictions.append(predictions)
                all_labels.append(test_labels)

            all_predictions = np.concatenate(all_predictions, axis=1)
            all_labels = np.concatenate(all_labels, axis=0)

        return jsonify({'status': 'Training step completed'})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
    

@app.route('/api/status', methods=['GET'])
def status():
    status = neural_net.get_status()
    return jsonify(status)


if __name__ == '__main__':
    app.run(debug=True)

