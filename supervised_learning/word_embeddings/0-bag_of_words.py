#!/usr/bin/env python3
"""Creates a bag-of-words embedding matrix."""
import numpy as np
import re


TOKEN_PATTERN = re.compile(r'\b\w\w+\b')


def _tokenize(sentence):
    """Tokenize a sentence into lowercase word tokens."""
    return TOKEN_PATTERN.findall(sentence.lower())


def _build_vocab(sentences):
    """Build a sorted vocabulary from a list of sentences."""
    vocab = set()
    for sentence in sentences:
        vocab.update(_tokenize(sentence))
    return sorted(vocab)


def bag_of_words(sentences, vocab=None):
    """Create a bag-of-words embedding matrix for a list of sentences."""
    if vocab is None:
        features = _build_vocab(sentences)
    else:
        features = [word.lower() for word in vocab]

    feature_index = {word: index for index, word in enumerate(features)}
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for sentence_index, sentence in enumerate(sentences):
        for token in _tokenize(sentence):
            if token in feature_index:
                embeddings[sentence_index, feature_index[token]] += 1

    return embeddings, features
