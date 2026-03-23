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

    def backward(self, dL_dZ, learning_rate, optimizer):
        for layer in reversed(self.layers):
            if isinstance(layer, ReLULayer):
                dL_dZ = layer.backward(dL_dZ)
            else:
                dL_dZ = layer.backward(dL_dZ, learning_rate, optimizer)

    def train(self, X, y, epochs, learning_rate, batch_size, optimizer):
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X = X[indices]
            y = y[indices]
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                output = self.forward(X_batch)  # through dense+relu layers only
                loss = self.loss_fn.forward(output, y_batch)  # loss layer separately
                dL_dZ = self.loss_fn.backward()
                self.backward(dL_dZ, learning_rate, optimizer)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def accuracy(self, X, y):
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        return np.mean(predictions == y)

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
    nn.train(X_train, y_train_one_hot, epochs=100, learning_rate=0.01, batch_size=32, optimizer='momentum')
    acc = nn.accuracy(X_train, y_train)
    print(f"Training Accuracy: {acc * 100:.2f}%")
    X_test, y_test = load_mnist('data/t10k-images-idx3-ubyte', 'data/t10k-labels-idx1-ubyte')
    test_acc = nn.accuracy(X_test, y_test)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
