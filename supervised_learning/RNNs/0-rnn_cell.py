#!/usr/bin/env python3
"""Defines a simple RNN cell."""
import numpy as np


class RNNCell:
    """Represents a simple RNN cell."""

    def __init__(self, i, h, o):
        """Initialize the RNN cell."""
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step."""
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = y / np.sum(y, axis=1, keepdims=True)
        return h_next, y
