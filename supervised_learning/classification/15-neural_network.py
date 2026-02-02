#!/usr/bin/env python3
"""Module for NeuralNetwork class"""
import numpy as np
import matplotlib.pyplot as plt


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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data

        Returns:
            The private attributes __A1 and __A2
        """
        # Calculate Z1 = W1·X + b1
        Z1 = np.matmul(self.__W1, X) + self.__b1
        # Apply sigmoid activation for hidden layer
        self.__A1 = 1 / (1 + np.exp(-Z1))
        
        # Calculate Z2 = W2·A1 + b2
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        # Apply sigmoid activation for output layer
        self.__A2 = 1 / (1 + np.exp(-Z2))
        
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (1, m) with correct labels
            A: numpy.ndarray with shape (1, m) with activated output

        Returns:
            The cost
        """
        m = Y.shape[1]
        # Calculate cost using logistic regression formula
        # Use 1.0000001 - A instead of 1 - A to avoid division by zero
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (1, m) with correct labels

        Returns:
            The network's prediction and the cost
        """
        # Get activated outputs
        _, A2 = self.forward_prop(X)
        # Get cost
        cost = self.cost(Y, A2)
        # Get predictions (1 if A2 >= 0.5, 0 otherwise)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (1, m) with correct labels
            A1: output of the hidden layer
            A2: predicted output
            alpha: learning rate
        """
        m = Y.shape[1]
        
        # Backpropagation for output layer
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # Backpropagation for hidden layer
        dZ1 = np.matmul(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = (1 / m) * np.matmul(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        
        # Update weights and biases
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (1, m) with correct labels
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
            A1, A2 = self.forward_prop(X)
            
            # Print and/or store cost at specified intervals
            if i == 0 or i == iterations or i % step == 0:
                cost = self.cost(Y, A2)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    costs.append(cost)
                    iterations_list.append(i)
            
            # Perform gradient descent (except after last iteration)
            if i < iterations:
                self.gradient_descent(X, Y, A1, A2, alpha)

        # Plot the graph if requested
        if graph:
            plt.plot(iterations_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        # Return evaluation after training
        return self.evaluate(X, Y)
