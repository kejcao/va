import os

prev = 0


def print_user_input(prompt, txt, erase=True, grayout=False):
    global prev

    cols, rows = os.get_terminal_size()

    if not erase:
        prev = 1

    while prev > 0:
        print('\r' + (' ' * cols), end='')
        if prev != 0:
            print('\033[F', end='')
        prev -= 1

    txt = prompt + txt
    prev = len(txt) // cols + 1

    if grayout:
        print('\033[90m', end='')
    while len(txt) > cols:
        txt = txt[:cols]
        print(txt)
    print(txt)
    if grayout:
        print('\033[0m', end='')
