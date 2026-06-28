#!/usr/bin/env python3
"""Defines the output layer of a bidirectional RNN cell."""
import numpy as np


class BidirectionalCell:
    """Represents a bidirectional RNN cell."""

    def __init__(self, i, h, o):
        """Initialize the bidirectional cell."""
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Calculate the hidden state in the forward direction."""
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """Calculate the hidden state in the backward direction."""
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(concat, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """Calculate all outputs for the RNN."""
        t, m, _ = H.shape
        Y = np.zeros((t, m, self.by.shape[1]))
        for step in range(t):
            y_linear = np.matmul(H[step], self.Wy) + self.by
            y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
            Y[step] = y / np.sum(y, axis=1, keepdims=True)
        return Y
