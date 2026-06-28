#!/usr/bin/env python3
"""Interactive QA bot over multiple reference documents."""

qa = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """Prompt the user and answer from the most relevant document.

    Args:
        corpus_path (str): Path to corpus of reference documents.
    """
    exits = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        question = input('Q: ')

        if question.lower() in exits:
            print('A: Goodbye')
            break

        reference = semantic_search(corpus_path, question)

        if reference is None:
            print('A: Sorry, I do not understand the question.')
            continue

        answer = qa(question, reference)
        if answer is None:
            print('A: Sorry, I do not understand the question.')
        else:
            print('A: {}'.format(answer))
