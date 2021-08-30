"""Module for representation and management of colors."""

from typing import Optional

import random

from nerte.values.color import Color


class RandomColorGenerator:
    """Generator for random colors."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._last_random_state: Optional[object] = None
        self.__reset()

    def __reset(self) -> None:
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

    def __iter__(self) -> "RandomColorGenerator":
        return self
