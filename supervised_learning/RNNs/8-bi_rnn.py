#!/usr/bin/env python3
"""Defines forward propagation for a bidirectional RNN."""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Perform forward propagation for a bidirectional RNN."""
    t, m, _ = X.shape
    _, h = h_0.shape
    o = bi_cell.by.shape[1]

    Hf = np.zeros((t + 1, m, h))
    Hb = np.zeros((t + 1, m, h))
    Hf[0] = h_0
    Hb[t] = h_t

    for step in range(t):
        Hf[step + 1] = bi_cell.forward(Hf[step], X[step])
        Hb[t - step - 1] = bi_cell.backward(Hb[t - step], X[t - step - 1])

    H = np.concatenate((Hf[1:], Hb[:-1]), axis=2)
    Y = bi_cell.output(H)
    return H, Y
