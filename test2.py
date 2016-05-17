import time


class A(object):
    def __init__(self):
        pass

    def run_big(self):
        x = 0
        start = time.time()
        for i in range(0, 100000000):
            x += 1
        print(time.time() - start)

A().run_big()