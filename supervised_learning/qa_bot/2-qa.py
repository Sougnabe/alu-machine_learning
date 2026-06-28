#!/usr/bin/env python3
"""Interactive QA loop over a single reference document."""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """Answer user questions from one reference text.

    Args:
        reference (str): Knowledge base text used to answer questions.
    """
    exits = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        question = input('Q: ')

        if question.lower() in exits:
            print('A: Goodbye')
            break

        answer = question_answer(question, reference)
        if answer is None:
            print('A: Sorry, I do not understand your question.')
        else:
            print('A: {}'.format(answer))
