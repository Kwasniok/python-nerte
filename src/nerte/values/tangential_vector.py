"""Module for representing a tangential vector."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector


class TangentialVector:
    # pylint: disable=R0903
    """
    Representation of a tangential vector.
    A tangential vector has a point (inside a manifold) from which the tangential
    space is obtained and a vector which resides within this space.
    """

    def __init__(
        self,
        point: Coordinates3D,
        vector: AbstractVector,
    ) -> None:
        self.point = point
        self.vector = vector

    def __repr__(self) -> str:
        return f"{self.vector} @ {self.point}"
