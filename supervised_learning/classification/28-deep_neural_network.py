#!/usr/bin/env python3
"""Module for DeepNeuralNetwork class"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers, activation='sig'):
        """
        Initialize a DeepNeuralNetwork instance

        Args:
            nx: number of input features
            layers: list representing number of nodes in each layer
            activation: type of activation function ('sig' or 'tanh')

        Raises:
            TypeError: If nx, layers, or activation is not valid
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

        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        # Initialize weights and biases for each layer
        for i in range(self.__L):
            if i == 0:
                # First layer connects to input
                self.__weights[f'W{i + 1}'] = (
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx))
            else:
                # Subsequent layers connect to previous layer
                self.__weights[f'W{i + 1}'] = (
                    np.random.randn(layers[i], layers[i - 1]) *
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

    @property
    def activation(self):
        """Getter for activation"""
        return self.__activation

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data

        Returns:
            The output of the neural network and the cache
        """
        # Store input in cache
        self.__cache['A0'] = X

        # Forward propagation through all layers
        for i in range(1, self.__L + 1):
            # Get weights and bias for current layer
            W = self.__weights[f'W{i}']
            b = self.__weights[f'b{i}']
            # Get activation from previous layer
            A_prev = self.__cache[f'A{i - 1}']

            # Calculate Z = W·A_prev + b
            Z = np.matmul(W, A_prev) + b

            # Apply activation based on layer and activation type
            if i == self.__L:
                # Softmax activation for output layer
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                # Apply specified activation for hidden layers
                if self.__activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))
                else:  # tanh
                    A = np.tanh(Z)

            # Store activation in cache
            self.__cache[f'A{i}'] = A

        # Return final activation and cache
        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (classes, m) with correct labels
            A: numpy.ndarray with shape (classes, m) with activated output

        Returns:
            The cost
        """
        m = Y.shape[1]
        # Calculate cost using cross-entropy for multiclass
        cost = -1 / m * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (classes, m) with correct labels

        Returns:
            The network's prediction and the cost
        """
        # Get activated output
        A, _ = self.forward_prop(X)
        # Get cost
        cost = self.cost(Y, A)
        # Get predictions (index of maximum value along axis 0)
        prediction = np.zeros_like(A)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            Y: numpy.ndarray with shape (classes, m) with correct labels
            cache: dictionary containing all intermediary values
            alpha: learning rate
        """
        m = Y.shape[1]

        # Backpropagation
        # Start with output layer
        A_L = cache[f'A{self.__L}']
        dZ = A_L - Y

        # Go backwards through the layers
        for i in range(self.__L, 0, -1):
            A_prev = cache[f'A{i - 1}']

            # Calculate gradients
            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            # Update weights and biases
            self.__weights[f'W{i}'] = self.__weights[f'W{i}'] - alpha * dW
            self.__weights[f'b{i}'] = self.__weights[f'b{i}'] - alpha * db

            # Calculate dZ for previous layer (if not at first layer)
            if i > 1:
                W = self.__weights[f'W{i}']
                A = cache[f'A{i - 1}']

                # Use appropriate derivative based on activation function
                if self.__activation == 'sig':
                    dZ = np.matmul(W.T, dZ) * A * (1 - A)
                else:  # tanh
                    dZ = np.matmul(W.T, dZ) * (1 - A * A)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the deep neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (classes, m) with correct labels
            iterations: number of iterations to train over
            alpha: learning rate
            verbose: whether to print information about training
            graph: whether to graph information about training
            step: number of iterations between printing/graphing

        Returns:
            The evaluation of the training data after iterations
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # Validate step if verbose or graph is True
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        # Lists to store costs for graphing
        costs = []
        iterations_list = []

        for i in range(iterations + 1):
            # Forward propagation
            A, cache = self.forward_prop(X)

            # Print and/or store cost at specified intervals
            if i == 0 or i == iterations or i % step == 0:
                cost = self.cost(Y, A)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    costs.append(cost)
                    iterations_list.append(i)

            # Perform gradient descent (except after last iteration)
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        # Plot the graph if requested
        if graph:
            plt.plot(iterations_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        # Return evaluation after training
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format

        Args:
            filename: file to which the object should be saved
        """
        # Add .pkl extension if not present
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object

        Args:
            filename: file from which the object should be loaded

        Returns:
            The loaded object, or None if file doesn't exist
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
