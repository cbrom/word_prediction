import readchar

while True:

    key = readchar.readkey()
    if key == readchar.key.ESC:
        break
    elif key == readchar.key.BACKSPACE:
        print('backspace')
    print(key)