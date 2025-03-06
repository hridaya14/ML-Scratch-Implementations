import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from NN import NeuralNetwork
from keras.api.datasets import mnist
import numpy as np

# Create Neural Network
nn = NeuralNetwork(784)
nn.addHiddenLayer(10)
nn.OutputLayer(10)

# Load MNIST dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Reshaping the dataset to have 784 features (28x28 images flattened to 1D array)
train_X = train_X.reshape((train_X.shape[0], 784))
test_X = test_X.reshape((test_X.shape[0], 784))

train_X = train_X / 255.0
test_X = test_X / 255.0


# Train the model
print("#### Training Model ####  ")
nn.train(train_X, train_y, 300)

print()

# Test the model
print("#### Testing model ####")
nn.test(test_X, test_y)