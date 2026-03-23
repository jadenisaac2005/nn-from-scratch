import numpy as np
from layers import DenseLayer, ReLULayer, SoftmaxCrossEntropy
from utils import load_mnist
class NeuralNetwork:
    def __init__(self, layers, loss_fn):
        self.layers = layers
        self.loss_fn = loss_fn

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dL_dZ, learning_rate):
        for layer in reversed(self.layers):
            if isinstance(layer, ReLULayer):
                dL_dZ = layer.backward(dL_dZ)
            else:
                dL_dZ = layer.backward(dL_dZ, learning_rate)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)  # through dense+relu layers only
            loss = self.loss_fn.forward(output, y)  # loss layer separately
            dL_dZ = self.loss_fn.backward()
            self.backward(dL_dZ, learning_rate)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

if __name__ == "__main__":
    # Load MNIST from local files (implement this function)
    X_train, y_train = load_mnist('data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte')
    # One-hot encode labels
    y_train_one_hot = np.zeros((y_train.size, 10))
    y_train_one_hot[np.arange(y_train.size), y_train] = 1

    # Build network
    layers = [
        DenseLayer(n_inputs=784, n_outputs=128, init='xavier_uniform', seed=42),
        ReLULayer(),
        DenseLayer(n_inputs=128, n_outputs=10, init='xavier_uniform', seed=43)
    ]
    loss_fn = SoftmaxCrossEntropy()
    nn = NeuralNetwork(layers, loss_fn)
    # Train
    nn.train(X_train, y_train_one_hot, epochs=100, learning_rate=0.01)
