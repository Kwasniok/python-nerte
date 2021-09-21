"""Module for representing rays."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, is_zero_vector
from nerte.values.tangential_vector import TangentialVector


class RaySegment:
    # pylint: disable=R0903
    """
    Representation of a straigt ray segment.
    A ray segment has a starting point (coordinate) and a direction
    (non-zero vector).

    NOTE: The length of the directional vector may be important or irrelevant
    depending on the usecase.
    NOTE: The length of the directional vector is never zero.
    """

    def __init__(
        self,
        tangential_vector: TangentialVector,
        is_finite: bool = True,
    ) -> None:
        if is_zero_vector(tangential_vector.vector):
            raise ValueError(
                "Cannot construct ray with zero vector as direction."
            )
        self.tangential_vector = tangential_vector
        self.is_finite = is_finite
        self.is_infinite = not is_finite

    def start(self) -> Coordinates3D:
        """Returns the starting point of the ray segment."""
        return self.tangential_vector.point

    def direction(self) -> AbstractVector:
        """Returns the direction/delta of the ray segment."""
        return self.tangential_vector.vector

    def __repr__(self) -> str:
        if self.is_finite:
            return f"RaySegment(tangential_vector={self.tangential_vector})"
        return f"RaySegment(tangential_vector={self.tangential_vector}, is_finite=False)"
