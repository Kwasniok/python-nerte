"""Module for representing discrete changes in ray segments."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.tangential_vector import TangentialVector
from nerte.values.ray_segment import RaySegment
from nerte.values.util.convert import coordinates_as_vector


class RaySegmentDelta:
    """Representation of the difference of two rays.

    NOTE: The difference of the starting coordinates of two rays is a tuple of
          arbitrary numbers. Most importantly the may not correspont to a valid
          coordinate of the underlying manifold!
    """

    def __init__(
        self, point_delta: AbstractVector, vector_delta: AbstractVector
    ) -> None:
        self.point_delta = point_delta
        self.vector_delta = vector_delta

    def __repr__(self) -> str:
        return f"RaySegmentDelta(point_delta={self.point_delta},vector_delta={self.vector_delta})"

    def __add__(self, other: "RaySegmentDelta") -> "RaySegmentDelta":
        return RaySegmentDelta(
            self.point_delta + other.point_delta,
            self.vector_delta + other.vector_delta,
        )

    def __sub__(self, other: "RaySegmentDelta") -> "RaySegmentDelta":
        return RaySegmentDelta(
            self.point_delta - other.point_delta,
            self.vector_delta - other.vector_delta,
        )

    def __neg__(self) -> "RaySegmentDelta":
        return RaySegmentDelta(-self.point_delta, -self.vector_delta)

    def __mul__(self, fac: float) -> "RaySegmentDelta":
        return RaySegmentDelta(self.point_delta * fac, self.vector_delta * fac)

    def __truediv__(self, fac: float) -> "RaySegmentDelta":
        return RaySegmentDelta(self.point_delta / fac, self.vector_delta / fac)


def ray_segment_as_delta(ray: RaySegment) -> RaySegmentDelta:
    """Converts a ray to a ray delta."""
    return RaySegmentDelta(
        coordinates_as_vector(ray.tangential_vector.point),
        ray.tangential_vector.vector,
    )


def add_ray_segment_delta(
    ray: RaySegment, ray_delta: RaySegmentDelta
) -> RaySegment:
    """Adds a ray coords to a ray."""
    return RaySegment(
        tangential_vector=TangentialVector(
            point=Coordinates3D(
                (
                    ray.tangential_vector.point[0] + ray_delta.point_delta[0],
                    ray.tangential_vector.point[1] + ray_delta.point_delta[1],
                    ray.tangential_vector.point[2] + ray_delta.point_delta[2],
                )
            ),
            vector=ray.tangential_vector.vector + ray_delta.vector_delta,
        )
    )
