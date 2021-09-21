"""Module for representing discrete changes in ray segments."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.tangential_vector import TangentialVector
from nerte.values.util.convert import coordinates_as_vector


class TangentialVectorDelta:
    """Representation of the difference of two tangential vectors.

    NOTE: The difference of the starting points of two tangential vectors is a
          tuple of arbitrary numbers. Most importantly the may not correspont to
          valid coordinates of the underlying manifold!
    """

    def __init__(
        self, point_delta: AbstractVector, vector_delta: AbstractVector
    ) -> None:
        self.point_delta = point_delta
        self.vector_delta = vector_delta

    def __repr__(self) -> str:
        return (
            f"TangentialVectorDelta("
            f"point_delta={self.point_delta}"
            f",vector_delta={self.vector_delta})"
        )

    def __add__(
        self, other: "TangentialVectorDelta"
    ) -> "TangentialVectorDelta":
        return TangentialVectorDelta(
            self.point_delta + other.point_delta,
            self.vector_delta + other.vector_delta,
        )

    def __sub__(
        self, other: "TangentialVectorDelta"
    ) -> "TangentialVectorDelta":
        return TangentialVectorDelta(
            self.point_delta - other.point_delta,
            self.vector_delta - other.vector_delta,
        )

    def __neg__(self) -> "TangentialVectorDelta":
        return TangentialVectorDelta(-self.point_delta, -self.vector_delta)

    def __mul__(self, fac: float) -> "TangentialVectorDelta":
        return TangentialVectorDelta(
            self.point_delta * fac, self.vector_delta * fac
        )

    def __truediv__(self, fac: float) -> "TangentialVectorDelta":
        return TangentialVectorDelta(
            self.point_delta / fac, self.vector_delta / fac
        )


def tangent_as_delta(tangent: TangentialVector) -> TangentialVectorDelta:
    """Converts a tangential vector to a tangential vector delta."""
    return TangentialVectorDelta(
        coordinates_as_vector(tangent.point),
        tangent.vector,
    )


def add_tangential_vector_delta(
    tangent: TangentialVector, tangent_delta: TangentialVectorDelta
) -> TangentialVector:
    """Adds a ray coords to a ray."""
    return TangentialVector(
        point=Coordinates3D(
            (
                tangent.point[0] + tangent_delta.point_delta[0],
                tangent.point[1] + tangent_delta.point_delta[1],
                tangent.point[2] + tangent_delta.point_delta[2],
            )
        ),
        vector=tangent.vector + tangent_delta.vector_delta,
    )
