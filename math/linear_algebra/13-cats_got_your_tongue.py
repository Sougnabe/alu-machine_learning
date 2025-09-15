#!/usr/bin/env python3
"""Module that provides a function to concatenate two matrices along a specific axis
"""

def np_cat(mat1, mat2, axis=0):
    """Concatenate two matrices along a specific axis (pure Python)"""
    if axis == 0:
        return mat1 + mat2
    elif axis == 1:
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:

        return None
