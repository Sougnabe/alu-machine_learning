#!/usr/bin/env python3
def np_slice(matrix, axes={}):
    """Slice a matrix along given axes"""
    slices = []
    for i in range(len(matrix.shape)):
        if i in axes:
            slices.append(slice(*axes[i]))
        else:
            slices.append(slice(None))
    return matrix[tuple(slices)]
