#!/usr/bin/env python3
"""Module for Neuron class"""
import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Initialize a Neuron instance

        Args:
            nx: The number of input features to the neuron

        Raises:
            TypeError: If nx is not an integer
            ValueError: If nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Initialize weights with random normal distribution
        # Shape is (1, nx) to match the expected output format
        self.W = np.random.randn(1, nx)

        # Initialize bias to 0
        self.b = 0

        # Initialize activated output to 0
        self.A = 0
