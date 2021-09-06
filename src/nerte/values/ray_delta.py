from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.ray import Ray
from nerte.values.util.convert import coordinates_as_vector


class RayDelta:
    """Representation of hthe difference of two rays.

    NOTE: The dirrerence of the starting coordinates of two rays is atuple of
          arbitrary numbers. Most importantly the may not correspont to a valid
          coordinate of the underlying manifold!
    """

    def __init__(
        self, start: AbstractVector, direction: AbstractVector
    ) -> None:
        self.start = start
        self.direction = direction

    def __repr__(self) -> str:
        return f"R𝚫(start={self.start},direction={self.direction})"

    def __add__(self, other: "RayDelta") -> "RayDelta":
        return RayDelta(
            self.start + other.start,
            self.direction + other.direction,
        )

    def __sub__(self, other: "RayDelta") -> "RayDelta":
        return RayDelta(
            self.start - other.start,
            self.direction - other.direction,
        )

    def __mul__(self, fac: float) -> "RayDelta":
        return RayDelta(self.start * fac, self.direction * fac)

    def __truediv__(self, fac: float) -> "RayDelta":
        return RayDelta(self.start / fac, self.direction / fac)


def ray_as_delta(ray: Ray) -> RayDelta:
    """Converts a ray to a ray delta."""
    return RayDelta(coordinates_as_vector(ray.start), ray.direction)


def add_ray_delta(ray: Ray, ray_delta: RayDelta) -> Ray:
    """Adds a ray delta to a ray."""
    return Ray(
        Coordinates3D(
            (
                ray.start[0] + ray_delta.start[0],
                ray.start[1] + ray_delta.start[1],
                ray.start[2] + ray_delta.start[2],
            )
        ),
        AbstractVector(
            (
                ray.direction[0] + ray_delta.direction[0],
                ray.direction[1] + ray_delta.direction[1],
                ray.direction[2] + ray_delta.direction[2],
            )
        ),
    )
