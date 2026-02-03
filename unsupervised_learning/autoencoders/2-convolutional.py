#!/usr/bin/env python3
"""Module for creating a convolutional autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder

    Args:
        input_dims: tuple of integers containing the dimensions of the model
                    input
        filters: list containing the number of filters for each convolutional
                 layer in the encoder, respectively
        latent_dims: tuple of integers containing the dimensions of the latent
                     space representation

    Returns:
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """
    # Encoder
    encoder_input = keras.Input(shape=input_dims)
    encoded = encoder_input

    for filter_size in filters:
        encoded = keras.layers.Conv2D(
            filter_size,
            (3, 3),
            activation='relu',
            padding='same'
        )(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)

    encoder = keras.Model(encoder_input, encoded)

    # Decoder
    decoder_input = keras.Input(shape=latent_dims)
    decoded = decoder_input

    for i, filter_size in enumerate(reversed(filters)):
        if i < len(filters) - 1:
            decoded = keras.layers.Conv2D(
                filter_size,
                (3, 3),
                activation='relu',
                padding='same'
            )(decoded)
            decoded = keras.layers.UpSampling2D((2, 2))(decoded)
        else:
            # Second to last convolution uses valid padding
            decoded = keras.layers.Conv2D(
                filter_size,
                (3, 3),
                activation='relu',
                padding='valid'
            )(decoded)
            decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    # Last convolution with sigmoid activation and no upsampling
    decoded = keras.layers.Conv2D(
        input_dims[-1],
        (3, 3),
        activation='sigmoid',
        padding='same'
    )(decoded)

    decoder = keras.Model(decoder_input, decoded)

    # Autoencoder
    auto_input = keras.Input(shape=input_dims)
    encoded_output = encoder(auto_input)
    decoded_output = decoder(encoded_output)
    auto = keras.Model(auto_input, decoded_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
