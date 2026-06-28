#!/usr/bin/env python3
"""RNN Decoder for machine translation."""
import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decodes target tokens using attention over encoder states."""

    def __init__(self, vocab, embedding, units, batch):
        """Class constructor."""
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Forward pass of the decoder for one time step."""
        context, _ = self.attention(s_prev, hidden_states)

        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)

        output, s = self.gru(x, initial_state=s_prev)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)

        return y, s
