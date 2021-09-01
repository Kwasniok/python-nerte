"""Module for Coordinate representation."""

# TODO: use typing.NewType instead?


class Coordinates:
    """
    Represenation of a three dimensional coordinate as a triple of real numbers.
    NOTE: Coordinates aren't vectors in the general case.
    """

    def __init__(self, x1: float, x2: float, x3: float) -> None:
        self._x: tuple[float, float, float] = (x1, x2, x3)

    def __repr__(self) -> str:
        return "C" + repr(self._x)

    def __getitem__(self, i: int) -> float:
        return self._x[i]


class Coordinates2D:
    """
    Represenation of a two dimensional coordinate as a pair of real numbers.
    NOTE: Coordinates aren't vectors in the general case.
    """

    def __init__(self, x1: float, x2: float) -> None:
        self._x: tuple[float, float] = (x1, x2)

    def __repr__(self) -> str:
        return "C" + repr(self._x)

    def __getitem__(self, i: int) -> float:
        return self._x[i]
