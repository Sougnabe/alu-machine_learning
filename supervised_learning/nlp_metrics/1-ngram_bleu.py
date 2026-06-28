#!/usr/bin/env python3
"""Calculates the n-gram BLEU score."""
import math


def _closest_ref_len(references, sent_len):
    """Returns reference length closest to sentence length."""
    return min((len(ref) for ref in references),
               key=lambda r_len: (abs(r_len - sent_len), r_len))


def _extract_ngrams(tokens, n):
    """Extracts n-grams and their counts from token sequence."""
    counts = {}
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def _clipped_precision(references, sentence, n):
    """Calculates clipped n-gram precision."""
    total = len(sentence) - n + 1
    if total <= 0:
        return 0

    sent_counts = _extract_ngrams(sentence, n)
    clipped = 0

    for ngram, count in sent_counts.items():
        max_ref_count = 0
        for ref in references:
            ref_counts = _extract_ngrams(ref, n)
            max_ref_count = max(max_ref_count, ref_counts.get(ngram, 0))
        clipped += min(count, max_ref_count)

    return clipped / total


def ngram_bleu(references, sentence, n):
    """Calculates the n-gram BLEU score for a sentence."""
    sent_len = len(sentence)
    if sent_len == 0:
        return 0

    precision = _clipped_precision(references, sentence, n)
    if precision == 0:
        return 0

    ref_len = _closest_ref_len(references, sent_len)
    bp = 1 if sent_len > ref_len else math.exp(1 - (ref_len / sent_len))

    return bp * precision
