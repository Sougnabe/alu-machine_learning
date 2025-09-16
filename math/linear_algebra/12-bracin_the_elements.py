#!/usr/bin/env python3
"""provides a function to perform element-wise operations on two matrices
"""


def np_elementwise(mat1, mat2):
    """Return sum, difference, product, and quotient"""
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return add, sub, mul, div
