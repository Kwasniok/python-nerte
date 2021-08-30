"""Module for representation and management of colors."""


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
