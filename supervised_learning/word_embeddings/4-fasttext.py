#!/usr/bin/env python3
"""Trains a gensim FastText model."""
import importlib


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Create and train a FastText model from sentences."""
    FastText = __import__('gensim.models', fromlist=['FastText']).FastText

    model = FastText(
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
