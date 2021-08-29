"""Module for representing faces."""

from nerte.geometry.coordinates import Coordinates


class Face:
    """"Represenation of a face as a triple of coordinates."""

    def __init__(
        self,
        c0: Coordinates,
        c1: Coordinates,
        c2: Coordinates,
    ):
        self._coords = (c0, c1, c2)

    def __repr__(self):
        return "F" + repr(self._coords)

    def __getitem__(self, i: int) -> Coordinates:
        return self._coords[i]

    def __iter__(self):
        return iter(self._coords)
