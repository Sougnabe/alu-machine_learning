#!/usr/bin/env python3
"""Calculates the unigram BLEU score."""
import math


def _closest_ref_len(references, sent_len):
    """Returns reference length closest to sentence length."""
    return min((len(ref) for ref in references),
               key=lambda r_len: (abs(r_len - sent_len), r_len))


def _precision_1(references, sentence):
    """Calculates clipped unigram precision."""
    sent_counts = {}
    for word in sentence:
        sent_counts[word] = sent_counts.get(word, 0) + 1

    clipped = 0
    for word, count in sent_counts.items():
        max_ref_count = 0
        for ref in references:
            ref_count = 0
            for token in ref:
                if token == word:
                    ref_count += 1
            max_ref_count = max(max_ref_count, ref_count)
        clipped += min(count, max_ref_count)

    return clipped / len(sentence)


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence."""
    sent_len = len(sentence)
    if sent_len == 0:
        return 0

    precision = _precision_1(references, sentence)
    if precision == 0:
        return 0

    ref_len = _closest_ref_len(references, sent_len)
    bp = 1 if sent_len > ref_len else math.exp(1 - (ref_len / sent_len))

    return bp * precision
