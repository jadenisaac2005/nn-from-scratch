import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from layers import DenseLayer


def test_classical_momentum_velocity_and_weights():
    """Fixed gradient g=1 for 3 steps: velocity must be g, 1.9g, 2.71g (beta=0.9)."""
    beta = 0.9
    learning_rate = 0.01

    layer = DenseLayer(n_inputs=1, n_outputs=1, init='xavier_uniform', seed=0)
    X = np.array([[1.0]])
    dL_dZ = np.array([[1.0]])
    layer.forward(X)  # sets layer.X so dL_dW = X.T @ dL_dZ == 1.0 each call

    expected_velocities = [1.0, 1.9, 2.71]
    W_prev = layer.W.copy()
    b_prev = layer.b.copy()

    for expected_v in expected_velocities:
        layer.backward(dL_dZ, learning_rate, optimizer='momentum', beta=beta)

        assert np.allclose(layer.v_W_momentum, expected_v), (
            f"expected v_W={expected_v}, got {layer.v_W_momentum}"
        )
        assert np.allclose(layer.v_b_momentum, expected_v), (
            f"expected v_b={expected_v}, got {layer.v_b_momentum}"
        )

        expected_W = W_prev - learning_rate * expected_v
        expected_b = b_prev - learning_rate * expected_v
        assert np.allclose(layer.W, expected_W), (
            f"expected W={expected_W}, got {layer.W}"
        )
        assert np.allclose(layer.b, expected_b), (
            f"expected b={expected_b}, got {layer.b}"
        )
        W_prev = layer.W.copy()
        b_prev = layer.b.copy()


if __name__ == '__main__':
    test_classical_momentum_velocity_and_weights()
    print("OK")
