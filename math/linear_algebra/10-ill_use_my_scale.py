#!/usr/bin/env python3
"""
Module that provides a function to get the shape of a matrix as a tuple
"""


def np_shape(matrix):
    """Return the shape of a matrix (nested lists) as a tuple"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return tuple(shape)
