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
- Mini-batch SGD
- SGD with Momentum (coming soon)
- Adam optimizer (coming soon)

## What's NOT Implemented

- CNN
- Batch normalization
- Dropout
- Transformers
- Framework-level abstraction
- C++

## Results

| Training Setup | Test Accuracy |
| --- | --- |
| Full-batch SGD | 68.59% |
| Mini-batch SGD (batch=32) | 97.82% |
| SGD + Momentum | coming soon |
| Adam | coming soon |

## Stack

- Python
- NumPy only — no PyTorch, no TensorFlow
