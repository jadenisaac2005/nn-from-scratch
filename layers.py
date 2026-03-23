import numpy as np

class DenseLayer:
    def __init__(self, n_inputs: int, n_outputs: int,
        init: str = 'xavier_uniform', seed: int = None):

        if seed is not None:
            np.random.seed(seed)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        if init == 'xavier_uniform':
            limit = np.sqrt(6.0 / (n_inputs + n_outputs))
            self.W = np.random.uniform(-limit, limit, size=(n_inputs, n_outputs))

        elif init == 'xavier_normal':
            std = np.sqrt(2.0 / (n_inputs + n_outputs))
            self.W = np.random.normal(0.0, std, size=(n_inputs, n_outputs))

        else:
            raise ValueError(f"Unknown init scheme: '{init}'. " f"Use 'xavier_uniform' or 'xavier_normal'.")

        self.b = np.zeros((1, n_outputs))

        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)
        self.t = 0

        self.v_W_momentum = np.zeros_like(self.W)
        self.v_b_momentum = np.zeros_like(self.b)

        self.X = None
        self.Z = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        assert X.shape[1] == self.n_inputs, (
            f"Input shape mismatch: expected (batch, {self.n_inputs}), got {X.shape}"
        )

        self.X = X
        self.Z = X @ self.W + self.b
        return self.Z

    def backward(self, dL_dZ: np.ndarray, learning_rate: float, optimizer='sgd', beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8) -> np.ndarray:
        dL_dW = np.dot(self.X.T, dL_dZ)
        dL_db = np.sum(dL_dZ, axis=0, keepdims=True)
        dL_dX = np.dot(dL_dZ, self.W.T)

        if optimizer == 'sgd':
            self.W -= learning_rate * dL_dW
            self.b -= learning_rate * dL_db

        elif optimizer == 'momentum':
            self.v_W_momentum = beta * self.v_W_momentum + (1 - beta) * dL_dW
            self.v_b_momentum = beta * self.v_b_momentum + (1 - beta) * dL_db
            self.W -= learning_rate * self.v_W_momentum
            self.b -= learning_rate * self.v_b_momentum

        elif optimizer == 'adam':
            # increment step
            self.t += 1

            # update first moment
            self.m_W = beta1 * self.m_W + (1 - beta1) * dL_dW
            self.m_b = beta1 * self.m_b + (1 - beta1) * dL_db

            # update second moment
            self.v_W = beta2 * self.v_W + (1 - beta2) * dL_dW ** 2
            self.v_b = beta2 * self.v_b + (1 - beta2) * dL_db ** 2

            # bias correction
            m_W_hat = self.m_W / (1 - beta1 ** self.t)
            m_b_hat = self.m_b / (1 - beta1 ** self.t)
            v_W_hat = self.v_W / (1 - beta2 ** self.t)
            v_b_hat = self.v_b / (1 - beta2 ** self.t)

            # update weights
            self.W -= learning_rate * m_W_hat / (np.sqrt(v_W_hat) + epsilon)
            self.b -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
        return dL_dX
class ReLULayer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(0, x)

    def backward(self, dL_dZ: np.ndarray) -> np.ndarray:
        dL_dX = dL_dZ * (self.x > 0)
        return dL_dX

class SoftmaxCrossEntropy:
    def forward(self, x, y):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.softmax = exps / np.sum(exps, axis=1, keepdims=True)
        self.y = y
        loss = -np.sum(y * np.log(self.softmax + 1e-15)) / x.shape[0]
        return loss
    def backward(self):
        batch_size = self.y.shape[0]
        dL_dZ = (self.softmax - self.y) / batch_size
        return dL_dZ


if __name__ == "__main__":
    # --- Sanity Check Suite ---
    np.random.seed(42)

    layer = DenseLayer(n_inputs=784, n_outputs=128, init='xavier_uniform', seed=42)

    # 1. Shape check
    assert layer.W.shape == (784, 128), "Weight shape wrong"
    assert layer.b.shape == (1, 128),   "Bias shape wrong"
    print("✓ Shapes correct")

    # 2. Xavier bound check
    limit = np.sqrt(6.0 / (784 + 128))
    assert np.all(layer.W >= -limit) and np.all(layer.W <= limit), "Weights outside Xavier bounds"
    print(f"✓ Weights within Xavier bounds ±{limit:.4f}")

    # 3. Variance check — should be close to 6 / (n_in + n_out) = ~0.0066
    expected_var = 6.0 / (784 + 128) / 3  # Var of Uniform(-a, a) = a²/3... wait, see note
    actual_var = np.var(layer.W)
    print(f"  Weight variance: {actual_var:.6f}  (expected ~{6.0 / (784 + 128) / 3:.6f})")

    # 4. Forward pass — batch of 32 samples
    X_dummy = np.random.randn(32, 784)
    Z = layer.forward(X_dummy)
    assert Z.shape == (32, 128), "Output shape wrong"
    print(f"✓ Forward pass output shape: {Z.shape}")

    # 5. Activation magnitude check — Z should be ~order 1, not exploding
    print(f"  Z mean: {Z.mean():.4f}, Z std: {Z.std():.4f}  (should be ~0 and ~1)")

    # 6. Cache check — X must be stored for backprop
    assert layer.X is not None, "Forward pass didn't cache input X"
    print("✓ Input X cached for backprop")
