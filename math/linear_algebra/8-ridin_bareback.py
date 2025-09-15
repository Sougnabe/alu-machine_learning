#!/usr/bin/env python3
"""
Module that provides a function to perform matrix multiplication
on two 2D matrices
"""


def mat_mul(mat1, mat2):
    """
    mat1 and mat2 to be multiplied
    """
    if len(mat1[0]) != len(mat2):
        return None

    result = []
    for row in mat1:
        new_row = []
        for col in zip(*mat2):
            new_row.append(sum(a * b for a, b in zip(row, col)))
        result.append(new_row)

    return result
