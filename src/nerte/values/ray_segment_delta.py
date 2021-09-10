"""Module for representing discrete changes in ray segments."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.ray_segment import RaySegment
from nerte.values.util.convert import coordinates_as_vector


class RaySegmentDelta:
    """Representation of the difference of two rays.

    NOTE: The difference of the starting coordinates of two rays is a tuple of
          arbitrary numbers. Most importantly the may not correspont to a valid
          coordinate of the underlying manifold!
    """

    def __init__(
        self, coords_delta: AbstractVector, velocity_delta: AbstractVector
    ) -> None:
        self.coords_delta = coords_delta
        self.velocity_delta = velocity_delta

    def __repr__(self) -> str:
        return f"R𝚫(coords_delta={self.coords_delta},velocity_delta={self.velocity_delta})"

    def __add__(self, other: "RaySegmentDelta") -> "RaySegmentDelta":
        return RaySegmentDelta(
            self.coords_delta + other.coords_delta,
            self.velocity_delta + other.velocity_delta,
        )

    def __sub__(self, other: "RaySegmentDelta") -> "RaySegmentDelta":
        return RaySegmentDelta(
            self.coords_delta - other.coords_delta,
            self.velocity_delta - other.velocity_delta,
        )

    def __neg__(self) -> "RaySegmentDelta":
        return RaySegmentDelta(-self.coords_delta, -self.velocity_delta)

    def __mul__(self, fac: float) -> "RaySegmentDelta":
        return RaySegmentDelta(
            self.coords_delta * fac, self.velocity_delta * fac
        )

    def __truediv__(self, fac: float) -> "RaySegmentDelta":
        return RaySegmentDelta(
            self.coords_delta / fac, self.velocity_delta / fac
        )


def ray_segment_as_delta(ray: RaySegment) -> RaySegmentDelta:
    """Converts a ray to a ray delta."""
    return RaySegmentDelta(coordinates_as_vector(ray.start), ray.direction)


def add_ray_segment_delta(
    ray: RaySegment, ray_delta: RaySegmentDelta
) -> RaySegment:
    """Adds a ray coords to a ray."""
    return RaySegment(
        Coordinates3D(
            (
                ray.start[0] + ray_delta.coords_delta[0],
                ray.start[1] + ray_delta.coords_delta[1],
                ray.start[2] + ray_delta.coords_delta[2],
            )
        ),
        AbstractVector(
            (
                ray.direction[0] + ray_delta.velocity_delta[0],
                ray.direction[1] + ray_delta.velocity_delta[1],
                ray.direction[2] + ray_delta.velocity_delta[2],
            )
        ),
    )
