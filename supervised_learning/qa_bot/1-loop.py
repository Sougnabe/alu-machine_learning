#!/usr/bin/env python3
"""Simple interactive loop for the QA bot."""


def main():
    """Run the user input loop."""
    exits = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        user = input('Q: ')
        if user.lower() in exits:
            print('A: Goodbye')
            break
        print('A:')


if __name__ == '__main__':
    main()
