#!/usr/bin/env python3
def np_slice(matrix, axes={}):
    """Slice a list of lists like numpy would"""
    slices = []
    for i in range(len(matrix[0])):
        slices.append(i)
    return [[row[j] for j in range(*axes.get(1, (0, len(row))))] 
            for row in matrix[axes.get(0, (0, len(matrix)))[0]:
                               axes.get(0, (0, len(matrix)))[1]]]
