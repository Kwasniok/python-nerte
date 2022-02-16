"""Module for representing faces."""

from collections.abc import Iterator

from nerte.values.coordinates import Coordinates3D


class Face:
    """ "Represenation of a face as a triple of coordinates."""

    def __init__(
        self,
        c0: Coordinates3D,
        c1: Coordinates3D,
        c2: Coordinates3D,
    ) -> None:
        # pylint: disable=C0103
        self._coords: tuple[Coordinates3D, Coordinates3D, Coordinates3D] = (
            c0,
            c1,
            c2,
        )

    def __repr__(self) -> str:
        return "F" + repr(self._coords)

    def __getitem__(self, i: int) -> Coordinates3D:
        return self._coords[i]

    def __iter__(self) -> Iterator[Coordinates3D]:
        return iter(self._coords)
