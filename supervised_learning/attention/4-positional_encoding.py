#!/usr/bin/env python3
"""Positional encoding for transformer models."""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates positional encoding vectors."""
    positions = np.arange(max_seq_len)[:, np.newaxis]
    dims = np.arange(dm)[np.newaxis, :]

    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / dm)
    angles = positions * angle_rates

    encoding = np.zeros((max_seq_len, dm))
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])

    return encoding
