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

| Optimizer | Learning Rate | Test Accuracy | First Epoch ≥ 97% |
| --- | --- | --- | --- |
| SGD | 0.01 | 97.90% | 26 |
| Momentum (β=0.9) | 0.01 | 98.07% | 4 |
| Adam | 0.001 | 97.91% | 2 |

Reproduce with:

```bash
python run_comparison.py
```

## Key Insights

- All three optimizers converge to ~97.9–98.1% test accuracy with correctly tuned learning rates
- Momentum and Adam reach 97% test accuracy within the first handful of epochs (4 and 2 respectively), while SGD takes 26 epochs to catch up
- SGD and Momentum training loss curves are monotonically decreasing every epoch; Adam's epoch-mean loss still fluctuates upward on ~40% of epochs late in training, which is genuine Adam noise rather than a logging artifact — see [results/history.json](results/history.json)

## What Broke

An earlier version implemented momentum in EMA form (v = βv + (1−β)g), which at the same learning rate behaves like plain SGD; replaced with classical momentum.

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
