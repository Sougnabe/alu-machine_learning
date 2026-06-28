#!/usr/bin/env python3
"""Question answering with a BERT model from TensorFlow Hub."""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


TOKENIZER = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad'
)
MODEL = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")


def question_answer(question, reference):
    """Find an answer span in reference for a given question.

    Args:
        question (str): The question to answer.
        reference (str): The reference context.

    Returns:
        str | None: Predicted answer span or None when no answer is found.
    """
    if not question or not reference:
        return None

    inputs = TOKENIZER.encode_plus(
        question,
        reference,
        max_length=512,
        truncation='only_second',
        return_tensors='tf'
    )

    input_ids = inputs['input_ids']
    input_mask = inputs['attention_mask']
    token_type_ids = inputs.get('token_type_ids')

    if token_type_ids is None:
        token_type_ids = tf.zeros_like(input_ids)

    outputs = MODEL([input_ids, input_mask, token_type_ids])
    start_logits, end_logits = outputs[0], outputs[1]

    start = int(tf.argmax(start_logits[0]).numpy())
    end = int(tf.argmax(end_logits[0]).numpy())

    null_score = float(start_logits[0][0] + end_logits[0][0])
    best_score = float(start_logits[0][start] + end_logits[0][end])

    if start == 0 or end == 0 or end < start or best_score <= null_score:
        return None

    tokens = input_ids[0][start:end + 1].numpy().tolist()
    answer = TOKENIZER.decode(tokens)
    answer = answer.strip()

    if not answer:
        return None

    return answer

