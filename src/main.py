import tensorflow as tf
import tensorflow_datasets as tfds
from arch.neural_network import NeuralNetwork
import numpy as np

"""
Nerual network with 3 layers of 16, 16, and 10 nodes
- inner layers use relu as the activation function
- outermost layer uses the softmax function
- cross entropy loss is used as the cost function
"""

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

    avg_loss = epoch_loss / num_batches

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
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

predicted_classes = np.argmax(all_predictions, axis=0)

accuracy = np.mean(predicted_classes == all_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")