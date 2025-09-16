#!/usr/bin/env python3
"""
Concatenates two numpy arrays along a specified axis.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Return the concatenation of mat1 and mat2 along the given axis"""
    return np.concatenate((mat1, mat2), axis=axis)
