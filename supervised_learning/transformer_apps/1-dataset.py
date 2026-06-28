#!/usr/bin/env python3
"""Dataset class with token encoding methods."""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Loads TED Portuguese to English data and creates tokenizers."""

    def __init__(self):
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
