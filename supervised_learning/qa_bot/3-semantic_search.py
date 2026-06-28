#!/usr/bin/env python3
"""Semantic document search using Universal Sentence Encoder."""
import os

import tensorflow as tf
import tensorflow_hub as hub


MODEL = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def semantic_search(corpus_path, sentence):
    """Find the most semantically similar document text in a corpus.

    Args:
        corpus_path (str): Path to folder containing reference documents.
        sentence (str): Query sentence.

    Returns:
        str | None: Full text of the most similar document, or None.
    """
    if not sentence or not os.path.isdir(corpus_path):
        return None

    files = sorted(os.listdir(corpus_path))

    documents = []
    for name in files:
        path = os.path.join(corpus_path, name)
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as file:
                documents.append(file.read())

    if not documents:
        return None

    doc_embeddings = MODEL(documents)
    sentence_embedding = MODEL([sentence])

    doc_embeddings = tf.math.l2_normalize(doc_embeddings, axis=1)
    sentence_embedding = tf.math.l2_normalize(sentence_embedding, axis=1)

    similarity = tf.matmul(sentence_embedding, doc_embeddings, transpose_b=True)
    best_index = int(tf.argmax(similarity[0]).numpy())

    return documents[best_index]
