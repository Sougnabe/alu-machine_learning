#!/usr/bin/env python3
"""Defines a gated recurrent unit cell."""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit cell."""

    def __init__(self, i, h, o):
        """Initialize the GRU cell."""
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step."""
        concat = np.concatenate((h_prev, x_t), axis=1)
        z = 1 / (1 + np.exp(-(np.matmul(concat, self.Wz) + self.bz)))
        r = 1 / (1 + np.exp(-(np.matmul(concat, self.Wr) + self.br)))
        concat_reset = np.concatenate((r * h_prev, x_t), axis=1)
        h_hat = np.tanh(np.matmul(concat_reset, self.Wh) + self.bh)
        h_next = (1 - z) * h_prev + z * h_hat
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = y / np.sum(y, axis=1, keepdims=True)
        return h_next, y
