#!/usr/bin/env python3
'''for sum'''

def summation_i_squared(n):
    '''returns the sum of the squares of all integers
    from 0 to n (inclusive)'''
    if not isinstance(n, int) or n < 0:
        return None
    return sum(i**2 for i in range(n + 1))