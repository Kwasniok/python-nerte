"""Module for representing rays."""

from nerte.geometry.coordinates import Coordinates
from nerte.geometry.vector import AbstractVector


class Ray:
    # pylint: disable=R0903
    """
    Representation of a ray.
    A ray has a starting point (coordinate) and a direction (vector).
    NOTE: The length of the directional vector may be important or irrelevant
    depending on the usecase.
    """

    def __init__(self, start: Coordinates, direction: AbstractVector):
        self.start = start
        self.direction = direction

    def __repr__(self):
        return "Ray({}, {})".format(repr(self.start), repr(self.direction))
