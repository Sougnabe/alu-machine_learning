#!/usr/bin/env python3
"""Module for creating training operation"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network

    Args:
        loss: loss of the network's prediction
        alpha: learning rate

    Returns:
        operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op
