#!/usr/bin/env python3
"""RNN Encoder for machine translation."""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Encodes an input sequence using an embedding and a GRU."""

    def __init__(self, vocab, embedding, units, batch):
        """Class constructor."""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """Initializes and returns the hidden state."""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Forward pass of the encoder."""
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
