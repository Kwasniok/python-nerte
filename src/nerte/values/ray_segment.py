"""Module for representing rays."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, is_zero_vector


class RaySegment:
    # pylint: disable=R0903
    """
    Representation of a straigt ray segment.
    A ray segment has a starting point (coordinate) and a direction (vector).

    NOTE: The length of the directional vector may be important or irrelevant
    depending on the usecase.
    NOTE: THe length of the directional vector is never zero.
    """

    def __init__(
        self,
        start: Coordinates3D,
        direction: AbstractVector,
        is_finite: bool = True,
    ) -> None:
        if is_zero_vector(direction):
            raise ValueError(
                "Cannot construct ray with zero vector as direction."
            )
        self.start = start
        self.direction = direction
        self.is_finite = is_finite
        self.is_infinite = not is_finite

    def __repr__(self) -> str:
        if self.is_finite:
            return (
                f"RaySegment(start={self.start}, direction={self.direction},)"
            )
        return f"RaySegment(start={self.start}, direction={self.direction}, is_finite=False)"