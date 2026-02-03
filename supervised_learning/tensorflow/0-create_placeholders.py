#!/usr/bin/env python3
"""Module for creating placeholders"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Creates two placeholders for the neural network

    Args:
        nx: number of feature columns in our data
        classes: number of classes in our classifier

    Returns:
        x: placeholder for the input data to the neural network
        y: placeholder for the one-hot labels for the input data
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y
