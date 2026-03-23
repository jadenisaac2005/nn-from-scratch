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

| Training Setup | Test Accuracy | Notes |
| --- | --- | --- |
| Full-batch SGD | 68.59% | Baseline |
| Mini-batch SGD (lr=0.01) | 97.82% | batch=32 |
| Adam (lr=0.01) | 97.05% | Unstable — loss spikes |
| Adam (lr=0.001) | 97.83% | Stable |
| Momentum (lr=0.01) | 97.87% | Best result |

## Key Insights

- Mini-batch vs full-batch is the single biggest accuracy jump (+29%)
- Adam with wrong learning rate performs worse than SGD
- Momentum edges out both SGD and Adam on this task
- All three optimizers converge to ~97.8-97.9% with correct tuning

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
