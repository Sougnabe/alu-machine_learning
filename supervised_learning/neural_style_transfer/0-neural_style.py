#!/usr/bin/env python3
"""Neural style transfer initialization and image scaling."""
import numpy as np
import tensorflow as tf


class NST:
    """Performs neural style transfer tasks."""

    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize an NST instance."""
        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if (not isinstance(alpha, (int, float)) or alpha < 0):
            raise TypeError("alpha must be a non-negative number")
        if (not isinstance(beta, (int, float)) or beta < 0):
            raise TypeError("beta must be a non-negative number")

        if not tf.executing_eagerly():
            tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Scale an image to a 512 px largest side and normalize pixels."""
        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        height, width, _ = image.shape
        largest_side = max(height, width)
        ratio = 512 / largest_side
        new_height = int(round(height * ratio))
        new_width = int(round(width * ratio))

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.image.resize_images(
            image, [new_height, new_width],
            method=tf.image.ResizeMethod.BICUBIC
        )
        image = image / 255.0
        return tf.expand_dims(image, axis=0)
