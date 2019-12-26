from threading import Thread
from queue import Queue


class WorkQueue(object):
    def __init__(self, capacity: int = 128, times: int = 3):
        assert times >= 1, "usually in_queue's size >= out_queue's size!"
        self.capacity = capacity
        self.in_queue = Queue(times * capacity)
        self.out_queue = Queue(capacity)
        self.transform()

    def get(self):
        tmp = []
        for i in range(self.capacity):
            tmp.append(self.out_queue.get())
        print(tmp)
        return tmp

    def run(self):
        def thred_func():
            cnt = 1
            while True:
                self.in_queue.put(cnt)
                cnt += 1

        t = Thread(target=thred_func, )
        t.setDaemon(True)
        t.start()

    def transform(self):
        def func():
            while True:
                self.out_queue.put(self.in_queue.get())

        t = Thread(target=func, )
        t.setDaemon(True)
        t.start()


# q = WorkQueue(capacity=10)
# q.run()
#
# for i in range(1000):
#     tmp = q.get()
#     print(tmp)

