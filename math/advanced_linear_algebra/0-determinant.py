#!/usr/bin/env python3
"""
Module that provides a function to calculate the determinant of a matrix
"""


def determinant(matrix):
    """Calculates the determinant of a square matrix"""
    if (not isinstance(matrix, list)
        or any(not isinstance(row, list) for row in matrix)):
       raise TypeError("matrix must be a list of lists")

        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    det = 0
    for c in range(n):
        submatrix = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * determinant(submatrix)
    return det
