#!/usr/bin/env python3
"""
Module that provides a function to perform element-wise operations on matrices
"""


def np_elementwise(mat1, mat2):
    """
    Perform element-wise addition, subtraction, multiplication, and division
    """
    # Si mat2 est un entier ou float, on le transforme en "matrice" de même taille que mat1
    if isinstance(mat2, (int, float)):
        mat2 = [[mat2 for _ in row] for row in mat1]

    add = [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(mat1, mat2)]
    sub = [[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(mat1, mat2)]
    mul = [[a * b for a, b in zip(r1, r2)] for r1, r2 in zip(mat1, mat2)]
    div = [[a / b for a, b in zip(r1, r2)] for r1, r2 in zip(mat1, mat2)]

    return (add, sub, mul, div)
