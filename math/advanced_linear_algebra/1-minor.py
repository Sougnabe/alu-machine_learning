#!/usr/bin/env python3
"""
Module that provides a function to calculate the minor matrix of a square matrix
"""

def determinant(matrix):
    """Helper function: calculates the determinant of a square matrix"""
    if matrix == [[]]:
        return 1
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    det = 0
    for c in range(n):
        submatrix = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1)**c) * matrix[0][c] * determinant(submatrix)
    return det


def minor(matrix):
    """Calculates the minor matrix of a square matrix"""
    if not isinstance(matrix, list) or any(not isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            submatrix = [row[:j] + row[j+1:] for k, row in enumerate(matrix) if k != i]
            minor_row.append(determinant(submatrix))
        minor_matrix.append(minor_row)

    return minor_matrix
