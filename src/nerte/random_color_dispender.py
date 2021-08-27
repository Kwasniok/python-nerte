import random


class RandomColorDispenser:
    def __init__(self, seed=0):
        self._seed = seed
        self._last_random_state = None
        self.__reset()

    def __reset(self):
        random.seed(self._seed)
        self._last_random_state = random.getstate()

    def __next__(self):
        random.setstate(self._last_random_state)
        color_r = random.randint(128, 255)
        color_g = random.randint(128, 255)
        color_b = random.randint(128, 255)
        color = (color_r, color_g, color_b)
        self._last_random_state = random.getstate()
        return color

    def __iter__(self):
        return self
