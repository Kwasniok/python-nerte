import random
from enum import Enum


class Color:
    def __init__(self, r, g, b):
        self.rgb = (r, g, b)


BLACK = Color(0, 0, 0)
GRAY = Color(128, 128, 128)


class RandomColorDispenser:
    def __init__(self, seed=0):
        self._seed = seed
        self._last_random_state = None
        self.__reset()

    def __reset(self):
        random.seed(self._seed)
        self._last_random_state = random.getstate()

    def __next__(self) -> Color:
        random.setstate(self._last_random_state)
        color_r = random.randint(128, 255)
        color_g = random.randint(128, 255)
        color_b = random.randint(128, 255)
        color = Color(color_r, color_g, color_b)
        self._last_random_state = random.getstate()
        return color

    def __iter__(self):
        return self
