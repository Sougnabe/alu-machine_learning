#!/usr/bin/env python3
"""
Module that provides a function to calculate the definiteness
of a square numpy.ndarray.
"""

import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    try:
        eigvals = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None

    pos = np.all(eigvals > 0)
    pos_semi = np.all(eigvals >= 0) and not pos
    neg = np.all(eigvals < 0)
    neg_semi = np.all(eigvals <= 0) and not neg
    indefinite = np.any(eigvals > 0) and np.any(eigvals < 0)

    if pos:
        return "Positive definite"
    if pos_semi:
        return "Positive semi-definite"
    if neg:
        return "Negative definite"
    if neg_semi:
        return "Negative semi-definite"
    if indefinite:
        return "Indefinite"

    return None
