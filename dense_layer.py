import numpy as np

class DenseLayer:
    """
    Fully-connected linear layer: Z = XW + b

    Stores forward-pass values needed for backprop later.
    """

    def __init__(self, n_inputs: int, n_outputs: int,
                    init: str = 'xavier_uniform', seed: int = None):
        """
        Args:
            n_inputs:   Number of input features (fan-in)
            n_outputs:  Number of neurons in this layer (fan-out)
            init:       Weight initialization scheme
            seed:       RNG seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # --- Weight initialization ---
        if init == 'xavier_uniform':
            limit = np.sqrt(6.0 / (n_inputs + n_outputs))
            self.W = np.random.uniform(-limit, limit, size=(n_inputs, n_outputs))

        elif init == 'xavier_normal':
            std = np.sqrt(2.0 / (n_inputs + n_outputs))
            self.W = np.random.normal(0.0, std, size=(n_inputs, n_outputs))

        else:
            raise ValueError(f"Unknown init scheme: '{init}'. "
                                f"Use 'xavier_uniform' or 'xavier_normal'.")

        # Biases always start at zero
        self.b = np.zeros((1, n_outputs))  # Shape (1, n_outputs) for broadcasting

        # Cache for backprop (populated during forward pass)
        self.X = None   # Input to this layer
        self.Z = None   # Pre-activation output

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: Z = XW + b

        Args:
            X: Input array of shape (batch_size, n_inputs)

        Returns:
            Z: Pre-activation output of shape (batch_size, n_outputs)
        """
        assert X.shape[1] == self.n_inputs, (
            f"Input shape mismatch: expected (batch, {self.n_inputs}), got {X.shape}"
        )

        self.X = X  # Cache — you'll need this for dL/dW in backprop
        self.Z = X @ self.W + self.b  # @ is np.matmul; clean and explicit
        return self.Z

    def backward(self, dL_dZ, learning_rate: float) -> np.ndarray:
        dL_dW = np.dot(self.X.T, dL_dZ)  # Shape (n_inputs, n_outputs)
        dL_db = np.sum(dL_dZ, axis=0, keepdims=True)  # Shape (1, n_outputs)
        dL_dX = np.dot(dL_dZ, self.W.T)  # Shape (batch_size, n_inputs)
        np.

        # update weights and biases
        self.W -= learning_rate * dL_dW
        self.b -= learning_rate * dL_db
        return dL_dX

class ReLULayer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x  # Cache for backprop
        return np.maximum(0, x)

    def backward(self, dL_dZ: np.ndarray) -> np.ndarray:
        dL_dX = dL_dZ * (self.x > 0)
        return dL_dX

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
    """

    **What you're looking for in output #5:** Z mean ≈ 0, Z std ≈ 0.8–1.2. If Xavier is working, you won't see std of 10 or 0.01 — that's the whole point.

    ---

    ## One Numerical Stability Note (For Later)

    The dense layer itself is numerically clean — it's just a matrix multiply. The danger zones are coming: **softmax** (exp overflow) and **cross-entropy** (log of near-zero). I'll flag those precisely when we get there. Plant this in your mind: you'll handle them by subtracting the max before exp, and you'll combine softmax + cross-entropy into a single numerically-stable function. Don't implement them separately and naively.

    ---

    ## What's Coming Next

    Once you've run those sanity checks and the layer feels solid, the natural build order is:

    1. **ReLU activation** — trivial forward, but the backward has a subtle mask operation
    2. **Softmax** — needs the numerical stability trick from day one
    3. **Cross-entropy loss** — and then immediately fuse it with softmax
    4. **Backprop through the dense layer** — this is where `self.X` and `self.Z` caches pay off

    The backprop equation for this layer is:
    ```
    dL/dW = X.T @ dL/dZ
    dL/db = sum(dL/dZ, axis=0, keepdims=True)
    dL/dX = dL/dZ @ W.T   ← this propagates the gradient to the previous layer
"""
