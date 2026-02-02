#!/usr/bin/env python3
"""Module for NeuralNetwork class"""
import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer"""

    def __init__(self, nx, nodes):
        """
        Initialize a NeuralNetwork instance

        Args:
            nx: number of input features
            nodes: number of nodes in the hidden layer

        Raises:
            TypeError: If nx or nodes is not an integer
            ValueError: If nx or nodes is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize private weights for hidden layer
        self.__W1 = np.random.randn(nodes, nx)
        # Initialize private bias for hidden layer
        self.__b1 = np.zeros((nodes, 1))
        # Initialize private activated output for hidden layer
        self.__A1 = 0
        # Initialize private weights for output neuron
        self.__W2 = np.random.randn(1, nodes)
        # Initialize private bias for output neuron
        self.__b2 = 0
        # Initialize private activated output for output neuron
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for W1"""
        return self.__W1

    @property
    def b1(self):
        """Getter for b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter for A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter for W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter for b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter for A2"""
        return self.__A2
