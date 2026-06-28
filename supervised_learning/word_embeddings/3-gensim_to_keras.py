#!/usr/bin/env python3
"""Converts a gensim Word2Vec model to a keras Embedding layer."""


def gensim_to_keras(model):
    """Return a trainable keras Embedding initialized from a gensim model."""
    return model.wv.get_keras_embedding(True)
