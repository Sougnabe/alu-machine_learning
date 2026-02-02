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

        # Initialize private weights with random normal distribution
        self.__W = np.random.randn(1, nx)
        # Initialize private bias to 0
        self.__b = 0
        # Initialize private activated output to 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data

        Returns:
            The activated output (A)
        """
        # Calculate Z = W·X + b
        Z = np.matmul(self.__W, X) + self.__b
        # Apply sigmoid activation function
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
