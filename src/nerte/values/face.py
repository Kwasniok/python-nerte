"""Module for representing faces."""

from collections.abc import Iterator

from nerte.values.coordinates import Coordinates


class Face:
    """"Represenation of a face as a triple of coordinates."""

    def __init__(
        self,
        c0: Coordinates,
        c1: Coordinates,
        c2: Coordinates,
    ) -> None:
        self._coords: tuple[Coordinates, Coordinates, Coordinates] = (
            c0,
            c1,
            c2,
        )

    def __repr__(self) -> str:
        return "F" + repr(self._coords)

    def __getitem__(self, i: int) -> Coordinates:
        return self._coords[i]

    def __iter__(self) -> Iterator[Coordinates]:
        return iter(self._coords)
