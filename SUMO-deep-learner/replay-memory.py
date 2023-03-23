import random

class Replay_memory:
    def __init__(self, size_max, size_min):
        self._samples = []
        self._size_max = size_max
        self._size_min = size_min


    def add_sample(self, sample):
        # adds state/action/reward sample into memory
        self._samples.append(sample)
        # if memory is full, remove oldest element
        if self._size_now() > self._size_max:
            self._samples.pop(0)


    def get_samples(self, n):
        # get samples from memory
        if self._size_now() < self._size_min:
            return []

        if n > self._size_now():
            return random.sample(self._samples, self._size_now())  # get all the samples
        else:
            return random.sample(self._samples, n)  # get "batch size" number of samples


    def _size_now(self):
        return len(self._samples)