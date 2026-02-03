#!/usr/bin/env python3
"""Module for creating a variational autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
                       layer in the encoder, respectively
        latent_dims: integer containing the dimensions of the latent space
                     representation

    Returns:
        encoder: the encoder model, which should output the latent
                 representation, the mean, and the log variance, respectively
        decoder: the decoder model
        auto: the full autoencoder model
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder_input

    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    # Mean and log variance for latent space
    z_mean = keras.layers.Dense(latent_dims, activation=None)(encoded)
    z_log_sigma = keras.layers.Dense(latent_dims, activation=None)(encoded)

    # Sampling function
    def sampling(args):
        """Reparameterization trick"""
        z_mean, z_log_sigma = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])

    encoder = keras.Model(encoder_input, [z, z_mean, z_log_sigma])

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = decoder_input

    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(
        decoded)
    decoder = keras.Model(decoder_input, decoder_output)

    # VAE
    auto_input = keras.Input(shape=(input_dims,))
    encoded_output, z_mean_out, z_log_sigma_out = encoder(auto_input)
    decoded_output = decoder(encoded_output)
    auto = keras.Model(auto_input, decoded_output)

    # VAE loss
    def vae_loss(y_true, y_pred):
        """VAE loss = reconstruction loss + KL divergence"""
        reconstruction_loss = keras.losses.binary_crossentropy(
            y_true, y_pred) * input_dims
        kl_loss = -0.5 * keras.backend.sum(
            1 + z_log_sigma_out - keras.backend.square(z_mean_out) -
            keras.backend.exp(z_log_sigma_out),
            axis=-1
        )
        return keras.backend.mean(reconstruction_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
