#!/usr/bin/env python3
"""Module for evaluating a neural network"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network

    Args:
        X: numpy.ndarray containing the input data to evaluate
        Y: numpy.ndarray containing the one-hot labels for X
        save_path: location to load the model from

    Returns:
        network's prediction, accuracy, and loss, respectively
    """
    with tf.Session() as sess:
        # Import the meta graph and restore weights
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        # Get tensors from collection
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        # Evaluate
        prediction, acc, cost = sess.run(
            [y_pred, accuracy, loss],
            feed_dict={x: X, y: Y}
        )

    return prediction, acc, cost
