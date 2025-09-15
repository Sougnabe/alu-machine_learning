#!/usr/bin/env python3
"""
documented
"""


def matrix_transpose(matrix):
    """
    documented
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
