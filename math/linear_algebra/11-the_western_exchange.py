#!/usr/bin/env python3
"""
Module that provides a function to get the transpose of a matrix
"""


def np_transpose(matrix):
    """Return the transpose of a matrix (list of lists)"""
    if not matrix:
        return []
    return [list(row) for row in zip(*matrix)]
