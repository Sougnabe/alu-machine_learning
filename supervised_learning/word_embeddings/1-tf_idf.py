#!/usr/bin/env python3
"""Creates a TF-IDF embedding matrix."""
import math
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


def _compute_idf(sentences, features):
    """Compute inverse document frequencies for the given features."""
    document_count = float(len(sentences))
    idf = {}

    for feature in features:
        document_frequency = 0
        for sentence in sentences:
            if feature in _tokenize(sentence):
                document_frequency += 1
        idf[feature] = math.log10(document_count / document_frequency) + 1

    return idf


def tf_idf(sentences, vocab=None):
    """Create a TF-IDF embedding matrix for a list of sentences."""
    if vocab is None:
        features = _build_vocab(sentences)
    else:
        features = [word.lower() for word in vocab]

    feature_index = {word: index for index, word in enumerate(features)}
    idf = _compute_idf(sentences, features)
    embeddings = np.zeros((len(sentences), len(features)), dtype=float)

    for sentence_index, sentence in enumerate(sentences):
        tokens = _tokenize(sentence)
        if not tokens:
            continue

        token_count = float(len(tokens))
        counts = {}
        for token in tokens:
            if token in feature_index:
                counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            tf = count / token_count
            embeddings[sentence_index, feature_index[token]] = tf * idf[token]

        norm = np.linalg.norm(embeddings[sentence_index])
        if norm > 0:
            embeddings[sentence_index] /= norm

    return embeddings, features
