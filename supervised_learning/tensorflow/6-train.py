#!/usr/bin/env python3
"""Module for training a neural network"""
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier

    Args:
        X_train: numpy.ndarray containing the training input data
        Y_train: numpy.ndarray containing the training labels
        X_valid: numpy.ndarray containing the validation input data
        Y_valid: numpy.ndarray containing the validation labels
        layer_sizes: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer
        alpha: learning rate
        iterations: number of iterations to train over
        save_path: designates where to save the model

    Returns:
        path where the model was saved
    """
    # Create placeholders
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Build the network
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculate loss and accuracy
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    # Create training operation
    train_op = create_train_op(loss, alpha)

    # Add to collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    # Initialize variables
    init = tf.global_variables_initializer()

    # Create saver
    saver = tf.train.Saver()

    # Start session
    with tf.Session() as sess:
        sess.run(init)

        # Training loop
        for i in range(iterations + 1):
            # Calculate training cost and accuracy
            train_cost, train_acc = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train}
            )

            # Calculate validation cost and accuracy
            valid_cost, valid_acc = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid}
            )

            # Print progress
            if i == 0 or i == iterations or i % 100 == 0:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_acc))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_acc))

            # Train (except after last iteration)
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        # Save the model
        save_path = saver.save(sess, save_path)

    return save_path
