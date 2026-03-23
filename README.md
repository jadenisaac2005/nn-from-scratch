# nn-from-scratch

## Project Goal

Implement a fully connected neural network from scratch (NumPy only) and analyze training dynamics under different optimizers and initialization strategies.

## What Will Be Implemented

- Dense layers
- ReLU
- Softmax
- Cross-entropy loss
- Backpropagation
- SGD, Momentum, Adam
- Xavier & He initialization
- MNIST training
- Experimental comparison

## What Will NOT Be Implemented

- CNN
- Batch norm
- Dropout
- Transformers
- Framework-level abstraction
- C++

## Results

| Training Setup | Test Accuracy |
| --- | --- |
| Full-batch SGD | 68.59% |
| Mini-batch SGD (batch=32) | 97.82% |
