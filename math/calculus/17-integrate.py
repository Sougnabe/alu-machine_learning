#!/usr/bin/env python3
"""
Calculate the integral of a polynomial.
"""

def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if not isinstance(poly, list) or not all(isinstance(x, (int, float)) for x in poly):
        return None
    if not isinstance(C, (int, float)):
        return None

    integral = [C]

    for i, coeff in enumerate(poly):
        new_coeff = coeff / (i + 1)
        if new_coeff.is_integer():
            new_coeff = int(new_coeff)
        integral.append(new_coeff)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
