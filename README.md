# nn-from-scratch

## Project Goal

Implement a fully connected neural network from scratch (NumPy only) and analyze training dynamics under different optimizers and initialization strategies.

## Architecture

784 → 128 → 10 (trained on MNIST)

## What's Implemented

- Dense layers with Xavier & He initialization
- ReLU activation
- Softmax + Cross-entropy loss
- Backpropagation (derived and implemented by hand)
- Mini-batch training with shuffle
- SGD optimizer
- SGD with Momentum optimizer
- Adam optimizer

## What's NOT Implemented

- CNN
- Batch normalization
- Dropout
- Transformers
- Framework-level abstraction
- C++

## Results

Architecture 784→128→10, 100 epochs, batch size 32, layer-init seeds (42, 43), shuffle seed 0.
Per-epoch training loss and test accuracy for every run are in [results/history.json](results/history.json); full config in [results/summary.json](results/summary.json).

| Optimizer | Learning Rate | Test Accuracy |
| --- | --- | --- |
| SGD | 0.01 | 97.90% |
| Momentum (β=0.9) | 0.01 | 97.87% |
| Adam | 0.001 | 97.91% |

Reproduce with:

```bash
python run_comparison.py
```

## Key Insights

- All three optimizers converge to ~97.9% test accuracy with correctly tuned learning rates
- Adam reaches near-zero training loss fastest but shows occasional late-training loss spikes (e.g. epoch 80)
- SGD and Momentum follow near-identical loss curves at this learning rate — see [results/history.json](results/history.json)

## Stack

- Python
- NumPy only — no PyTorch, no TensorFlow

## Project Structure

```
nn-from-scratch/
├── layers.py        # DenseLayer, ReLULayer, SoftmaxCrossEntropy
├── train.py         # NeuralNetwork class + training loop
├── utils.py         # MNIST data loader
└── data/            # MNIST binary files
```
