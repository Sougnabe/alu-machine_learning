#!/usr/bin/env python3
"""Module for DeepNeuralNetwork class"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """
        Initialize a DeepNeuralNetwork instance

        Args:
            nx: number of input features
            layers: list representing number of nodes in each layer

        Raises:
            TypeError: If nx or layers is not the correct type
            ValueError: If nx or layers is not valid
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        
        # Validate all layers are positive integers
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialize weights and biases for each layer
        for i in range(self.__L):
            if i == 0:
                # First layer connects to input
                self.__weights[f'W{i + 1}'] = (np.random.randn(layers[i], nx) *
                                                np.sqrt(2 / nx))
            else:
                # Subsequent layers connect to previous layer
                self.__weights[f'W{i + 1}'] = (np.random.randn(layers[i],
                                               layers[i - 1]) *
                                                np.sqrt(2 / layers[i - 1]))
            # Initialize biases to 0
            self.__weights[f'b{i + 1}'] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights
