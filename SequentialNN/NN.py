# basic 3 layer neural network implementation

import numpy as np
from keras.api.datasets import mnist


# Helper Functions
def init_Weights(prev_nodes: int, curr_nodes: int):
    w = np.random.uniform(-1, 1, (curr_nodes, prev_nodes))
    b = np.random.uniform(-1, 1, (curr_nodes, 1))
    return w, b

def Relu(input: np.ndarray, derivative=False):
    """ Function to compute ReLu and Derivative of relu"""
    if derivative:
        return np.where(input > 0, 1, 0)
    return np.maximum(0, input)

def softmax(x):
    """
    Compute softmax values for each set of values in x with numerical stability.
    - Subtracted maximum value from each element for stability ensuring exponents stay within reasonable limits
    """
    x = x - np.max(x, axis=0, keepdims=True)  
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=0, keepdims=True)

def one_hot_encode(y, num_classes=10):
    """ OneHot encoding the actual labels"""
    one_hot_y = np.zeros((len(y), num_classes))
    one_hot_y[np.arange(len(y)), y] = 1
    return one_hot_y


class NeuralNetwork:
    def __init__(self, input_size: int):
        self.layers = []
        self.output = None
        self.input_size = input_size
        self.z_values = []
        self.activations = []

    # Adding a Hidden layer logic
    def addHiddenLayer(self, num_nodes=10):
        if len(self.layers) == 0:
            prev_nodes = self.input_size
        else:
            prev_nodes = self.layers[-1][0].shape[0]

        weights, bias = init_Weights(prev_nodes, num_nodes)
        self.layers.append((weights, bias))

    # Defining the output layer logic
    def OutputLayer(self, num_nodes: int):
        prev_nodes = self.layers[-1][0].shape[0]
        weight, bias = init_Weights(prev_nodes, num_nodes)
        self.OutputWeights = (weight, bias)

    def forward(self, X: np.ndarray):
        """
            Standard forward logic that performs dot multiplication and then applies the necessary activation function.
        """
        self.activations = [X]
        self.z_values = []

        outputs = X

        # Hidden layers use ReLU
        for weights, bias in self.layers:
            z = np.dot(weights, outputs) + bias
            self.z_values.append(z)
            outputs = Relu(z)
            self.activations.append(outputs)

        # Output layer uses softmax
        output_sums = self.OutputSum = np.dot(self.OutputWeights[0], outputs) + self.OutputWeights[1]
        output_activations = self.OutputActivation = softmax(output_sums)  # Keep softmax for classification

        return output_activations

    def backPropogate(self, y, lr):
        """
            Function to implement backpropogation algorithm.
            - Uses chain rule to find gradient of cost function with respect to weights of each layer
        """
        # Compute loss
        loss = np.sum((y - self.OutputActivation)**2)

        # Updating output layer weights
        const = 2 * (self.OutputActivation - y)
        dW = np.dot(const, self.activations[-1].T)
        dB = np.sum(const, axis=1, keepdims=True)

        W, B = self.OutputWeights
        self.OutputWeights = (W - lr * dW, B - lr * dB)

        # Updating hidden layer weights
        for idx in range(len(self.layers) - 1, -1, -1):
            dW = np.dot(const, self.activations[idx].T)
            dB = np.sum(const, axis=1, keepdims=True)
            for i in range(idx - 1, -1, -1):
                if i == 0:
                    dB *= Relu(self.z_values[i], derivative=True)
                    dW = np.dot(dB, self.activations[i].T)
                else:
                    dW = np.dot(Relu(self.z_values[idx], derivative=True) * dB, self.activations[i - 1].T)
                    dB = dW
            # updating weight for specific hidden layer
            W, B = self.layers[idx]
            self.layers[idx] = (W - lr * dW, B - lr * dB)

        return loss

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, lr=0.001):
        num_classes = 10  # Ensure num_classes is set to 10
        y_encoded = one_hot_encode(y, num_classes).T

        for epoch in range(epochs):
            total_loss = 0
            for idx in range(X.shape[0]):
                input = X[idx].reshape(-1, 1)
                self.forward(input)

                total_loss += self.backPropogate(y_encoded[:, idx].reshape(-1, 1), lr)

            if epoch % 50 == 0:
                print(f"Epoch {epoch}, loss: {total_loss / X.shape[0]:.4f}")

    def predict(self, X: np.ndarray):
        predictions = []
        for x in X:
            output = self.forward(x.reshape(-1, 1))
            predictions.append(np.argmax(output))
        return np.array(predictions)

    def test(self, X: np.ndarray, y: np.ndarray):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

