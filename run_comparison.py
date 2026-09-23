import json
import time
from pathlib import Path

import numpy as np

from layers import DenseLayer, ReLULayer, SoftmaxCrossEntropy
from train import NeuralNetwork
from utils import load_mnist

LAYER_SEEDS = (42, 43)
SHUFFLE_SEED = 0
EPOCHS = 100
BATCH_SIZE = 32

OPTIMIZER_CONFIGS = {
    'sgd':      {'learning_rate': 0.01},
    'momentum': {'learning_rate': 0.01, 'beta': 0.9},
    'adam':     {'learning_rate': 0.001},
}


def build_network():
    layers = [
        DenseLayer(n_inputs=784, n_outputs=128, init='xavier_uniform', seed=LAYER_SEEDS[0]),
        ReLULayer(),
        DenseLayer(n_inputs=128, n_outputs=10, init='xavier_uniform', seed=LAYER_SEEDS[1]),
    ]
    return NeuralNetwork(layers, SoftmaxCrossEntropy())


def run(optimizer, learning_rate, X_train, y_train_one_hot, X_test, y_test, epochs):
    nn = build_network()
    return nn.train(
        X_train, y_train_one_hot,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=BATCH_SIZE,
        optimizer=optimizer,
        seed=SHUFFLE_SEED,
        X_test=X_test,
        y_test=y_test,
    )


def main():
    X_train, y_train = load_mnist('data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte')
    y_train_one_hot = np.zeros((y_train.size, 10))
    y_train_one_hot[np.arange(y_train.size), y_train] = 1
    X_test, y_test = load_mnist('data/t10k-images-idx3-ubyte', 'data/t10k-labels-idx1-ubyte')

    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    history_all = {}
    final_test_acc = {}

    for name, cfg in OPTIMIZER_CONFIGS.items():
        print(f"=== {name} ===")
        t0 = time.time()
        history = run(name, cfg['learning_rate'], X_train, y_train_one_hot, X_test, y_test, EPOCHS)
        elapsed = time.time() - t0
        print(f"{name} done in {elapsed:.1f}s, final test_acc={history['test_acc'][-1] * 100:.2f}%")
        history_all[name] = history
        final_test_acc[name] = history['test_acc'][-1]

    summary = {
        'config': {
            'architecture': '784-128-10',
            'layer_seeds': list(LAYER_SEEDS),
            'shuffle_seed': SHUFFLE_SEED,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'optimizers': OPTIMIZER_CONFIGS,
        },
        'final_test_acc': final_test_acc,
    }

    with open(results_dir / 'history.json', 'w') as f:
        json.dump(history_all, f, indent=2)
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("Saved results/history.json and results/summary.json")


if __name__ == '__main__':
    main()
