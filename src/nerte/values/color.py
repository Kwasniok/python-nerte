"""Module for representation and management of colors."""


class Color:
    # pylint: disable=R0903
    """Represenation of a color."""

    def __init__(self, r: int, g: int, b: int) -> None:
        # pylint: disable=C0103
        self.rgb: tuple[int, int, int] = (r, g, b)

    def __repr__(self) -> str:
        return f"Color(r={self.rgb[0]},g={self.rgb[1]},b={self.rgb[2]})"


class Colors:
    # pylint: disable=R0903
    """Bundle of color constants."""

    BLACK = Color(0, 0, 0)
    GRAY = Color(128, 128, 128)
    WHITE = Color(255, 255, 255)
