#!/usr/bin/env python3
def np_elementwise(mat1, mat2):
    """Return element-wise sum, difference, product, and quotient"""
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return add, sub, mul, div
