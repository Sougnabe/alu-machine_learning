#!/usr/bin/env python3
"""Module that provides a function to perform matrix multiplication
"""

def np_matmul(mat1, mat2):
    """Performs matrix multiplication of two 2D matrices (pure Python)"""
    if len(mat1[0]) != len(mat2):
        return None
    return [
        [sum(a * b for a, b in zip(row, col)) for col in zip(*mat2)]
        for row in mat1
    ]
