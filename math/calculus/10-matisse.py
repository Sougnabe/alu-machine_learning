#!/usr/bin/env python3
'''for derivative of polynomial'''
def poly_derivative(poly):

    if not isinstance(poly, list) or len(poly) == 0:
        return None


    if len(poly) == 1:
        return [0]

    deriv = [coeff * i for i, coeff in enumerate(poly)][1:]
    return deriv
