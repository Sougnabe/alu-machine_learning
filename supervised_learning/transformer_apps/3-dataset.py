#!/usr/bin/env python3
"""Dataset class with full tf.data pipeline."""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Loads and prepares translation data for transformer training."""

    def __init__(self, batch_size, max_len):
        """Class constructor."""
        examples, _ = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            with_info=True,
            as_supervised=True
        )

        self.data_train = examples['train']
        self.data_valid = examples['validation']
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        self.data_train = self.data_train.filter(
            lambda pt, en: self.filter_max_len(pt, en, max_len)
        )
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
        )

        self.data_valid = self.data_valid.filter(
            lambda pt, en: self.filter_max_len(pt, en, max_len)
        )
        self.data_valid = self.data_valid.padded_batch(batch_size)
        self.data_validate = self.data_valid

    def tokenize_dataset(self, data):
        """Creates subword tokenizers for Portuguese and English."""
        pt_corpus = (pt.numpy() for pt, _ in data)
        en_corpus = (en.numpy() for _, en in data)

        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            pt_corpus,
            target_vocab_size=2 ** 15
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            en_corpus,
            target_vocab_size=2 ** 15
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encodes a translation pair with start and end tokens."""
        pt_tokens = [self.tokenizer_pt.vocab_size]
        pt_tokens += self.tokenizer_pt.encode(pt.numpy().decode('utf-8'))
        pt_tokens += [self.tokenizer_pt.vocab_size + 1]

        en_tokens = [self.tokenizer_en.vocab_size]
        en_tokens += self.tokenizer_en.encode(en.numpy().decode('utf-8'))
        en_tokens += [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """TensorFlow wrapper around encode."""
        pt_tokens, en_tokens = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens

    @staticmethod
    def filter_max_len(pt, en, max_len):
        """Filters examples by tokenized sentence lengths."""
        return tf.logical_and(tf.size(pt) <= max_len, tf.size(en) <= max_len)
