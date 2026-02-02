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

        # Initialize weights for hidden layer with random normal distribution
        self.W1 = np.random.randn(nodes, nx)
        # Initialize bias for hidden layer to 0's
        self.b1 = np.zeros((nodes, 1))
        # Initialize activated output for hidden layer to 0
        self.A1 = 0
        # Initialize weights for output neuron with random normal distribution
        self.W2 = np.random.randn(1, nodes)
        # Initialize bias for output neuron to 0
        self.b2 = 0
        # Initialize activated output for output neuron to 0
        self.A2 = 0
