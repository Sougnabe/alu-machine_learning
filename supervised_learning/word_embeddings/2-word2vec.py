#!/usr/bin/env python3
"""Trains a gensim Word2Vec model."""
import importlib


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Create and train a Word2Vec model from sentences."""
    Word2Vec = __import__('gensim.models', fromlist=['Word2Vec']).Word2Vec

    model = Word2Vec(
        sentences=sentences,
        size=size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=0 if cbow else 1,
        iter=iterations,
        seed=seed,
        workers=workers
    )
    return model
