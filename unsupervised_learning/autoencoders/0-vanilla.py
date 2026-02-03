#!/usr/bin/env python3
"""Module for creating a vanilla autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder

    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
                       layer in the encoder, respectively
        latent_dims: integer containing the dimensions of the latent space
                     representation

    Returns:
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder_input

    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    encoder = keras.Model(encoder_input, latent)

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = decoder_input

    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(
        decoded)
    decoder = keras.Model(decoder_input, decoder_output)

    # Autoencoder
    auto_input = keras.Input(shape=(input_dims,))
    encoded_output = encoder(auto_input)
    decoded_output = decoder(encoded_output)
    auto = keras.Model(auto_input, decoded_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
