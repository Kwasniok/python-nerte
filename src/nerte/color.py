"""Module for representation and management of colors."""

from typing import Optional

import random


class Color:
    # pylint: disable=R0903
    """Represenation of a color."""

    def __init__(self, r: int, g: int, b: int) -> None:
        self.rgb: tuple[int, int, int] = (r, g, b)


class Colors:
    # pylint: disable=R0903
    """Bundle of color constants."""

    BLACK = Color(0, 0, 0)
    GRAY = Color(128, 128, 128)
    WHITE = Color(255, 255, 255)


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
