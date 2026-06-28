#!/usr/bin/env python3
"""Neural style transfer with variational cost."""
import numpy as np
import tensorflow as tf


class NST:
    """Performs neural style transfer tasks."""

    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1,
                 var=10):
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
        if (not isinstance(var, (int, float)) or var < 0):
            raise TypeError("var must be a non-negative number")

        if not tf.executing_eagerly():
            tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.var = var
        self.model = None
        self.gram_style_features = None
        self.content_feature = None
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Scale an image to a 512 px largest side and normalize pixels."""
        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )
        height, width, _ = image.shape
        ratio = 512 / max(height, width)
        new_height = int(round(height * ratio))
        new_width = int(round(width * ratio))
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.image.resize_images(
            image, [new_height, new_width],
            method=tf.image.ResizeMethod.BICUBIC
        )
        image = image / 255.0
        return tf.expand_dims(image, axis=0)

    def load_model(self):
        """Create the VGG19-based model used to calculate costs."""
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(layer).output for layer in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)
        self.model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculate the gram matrix of a rank-4 tensor."""
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                input_layer.shape.ndims != 4):
            raise TypeError("input_layer must be a tensor of rank 4")
        shape = tf.shape(input_layer)
        height = shape[1]
        width = shape[2]
        channels = shape[3]
        features = tf.reshape(input_layer, (height * width, channels))
        gram = tf.matmul(features, features, transpose_a=True)
        gram = gram / tf.cast(height * width * channels, tf.float32)
        return tf.expand_dims(gram, axis=0)

    def generate_features(self):
        """Extract style and content features from the reference images."""
        outputs = self.model(self.style_image)
        self.gram_style_features = [self.gram_matrix(output)
                                    for output in outputs[:-1]]
        self.content_feature = self.model(self.content_image)[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Calculate the style cost for a single layer."""
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                style_output.shape.ndims != 4):
            raise TypeError("style_output must be a tensor of rank 4")
        channels = style_output.shape[-1]
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape.ndims != 3 or
                gram_target.shape[1] != channels or
                gram_target.shape[2] != channels):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {c}, {c}]".format(
                    c=channels
                )
            )
        gram_style = self.gram_matrix(style_output)
        shape = tf.shape(style_output)
        height = tf.cast(shape[1], tf.float32)
        width = tf.cast(shape[2], tf.float32)
        channels = tf.cast(shape[3], tf.float32)
        normalization = 4.0 * tf.pow(channels, 2) * tf.pow(height * width, 2)
        return tf.reduce_sum(tf.square(gram_style - gram_target)) / normalization

    def style_cost(self, style_outputs):
        """Calculate the style cost for the generated image."""
        if (not isinstance(style_outputs, list) or
                len(style_outputs) != len(self.style_layers)):
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    len(self.style_layers)
                )
            )
        style_cost = 0
        for style_output, gram_target in zip(style_outputs,
                                             self.gram_style_features):
            style_cost += self.layer_style_cost(style_output, gram_target)
        return style_cost / len(self.style_layers)

    def content_cost(self, content_output):
        """Calculate the content cost for the generated image."""
        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != self.content_feature.shape):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(
                    self.content_feature.shape
                )
            )
        return 0.5 * tf.reduce_sum(
            tf.square(content_output - self.content_feature)
        )

    @staticmethod
    def variational_cost(generated_image):
        """Calculate the variational cost for the generated image."""
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape.ndims != 4):
            raise TypeError("generated_image must be a tensor of rank 4")
        return tf.image.total_variation(generated_image)

    def total_cost(self, generated_image):
        """Calculate the total cost for the generated image."""
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != self.content_image.shape):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    self.content_image.shape
                )
            )
        outputs = self.model(generated_image)
        style_outputs = outputs[:-1]
        content_output = outputs[-1]
        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        J_var = self.variational_cost(generated_image)
        J_total = self.alpha * J_content + self.beta * J_style + self.var * J_var
        return J_total, J_content, J_style, J_var

    def compute_grads(self, generated_image):
        """Calculate gradients for a generated image."""
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != self.content_image.shape):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    self.content_image.shape
                )
            )
        with tf.GradientTape() as tape:
            if not isinstance(generated_image, tf.Variable):
                tape.watch(generated_image)
            J_total, J_content, J_style, J_var = self.total_cost(generated_image)
        gradients = tape.gradient(J_total, generated_image)
        return gradients, J_total, J_content, J_style, J_var

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Generate the neural style transferred image."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError(
                    "iterations must be positive and less than iterations"
                )
        if not isinstance(lr, (int, float)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.Variable(self.content_image)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr, beta1=beta1, beta2=beta2
        )
        best_image = tf.identity(generated_image)
        best_cost = float('inf')

        for i in range(iterations + 1):
            gradients, J_total, J_content, J_style, J_var = self.compute_grads(
                generated_image
            )
            current_cost = J_total.numpy()
            if current_cost < best_cost:
                best_cost = current_cost
                best_image = tf.identity(generated_image)
            if step is not None and (i % step == 0 or i == iterations):
                print(
                    "Cost at iteration {}: {}, content {}, style {}, var {}".format(
                        i, J_total.numpy(), J_content.numpy(),
                        J_style.numpy(), J_var.numpy()
                    )
                )
            if i < iterations:
                optimizer.apply_gradients([(gradients, generated_image)])
                generated_image.assign(tf.clip_by_value(generated_image, 0, 1))
        return best_image.numpy()[0], best_cost
