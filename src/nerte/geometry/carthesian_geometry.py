"""Module for representing an euclidean geometry in carthesian coordinates."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, normalized, is_zero_vector
from nerte.values.face import Face
from nerte.values.ray_segment import RaySegment
from nerte.values.intersection_info import IntersectionInfo
from nerte.values.util.convert import coordinates_as_vector
from nerte.geometry.geometry import Geometry, intersection_ray_depth


class CarthesianGeometry(Geometry):
    """Represenation of the euclidean geometry in Carthesian coordinates."""

    class Ray(Geometry.Ray):
        """Represenation of a ray in euclidean geometry in Carthesian coordinates."""

        def __init__(self, start: Coordinates3D, direction: AbstractVector):
            if is_zero_vector(direction):
                raise ValueError(
                    "Cannot construct carthesian ray with zero vector as direction."
                )
            direction = normalized(direction)
            self._segment = RaySegment(
                start=start, direction=direction, is_finite=False
            )

        def intersection_info(self, face: Face) -> IntersectionInfo:
            ray_depth = intersection_ray_depth(ray=self._segment, face=face)
            # no length factor required since segment is normalized
            return IntersectionInfo(ray_depth=ray_depth)

        def as_segment(self) -> RaySegment:
            """Returns ray converted to a ray segment."""
            return self._segment

    def __init__(self) -> None:
        pass

    def is_valid_coordinate(self, coordinates: Coordinates3D) -> bool:
        return True

    def ray_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> Ray:
        vec_s = coordinates_as_vector(start)
        vec_t = coordinates_as_vector(target)
        return CarthesianGeometry.Ray(start=start, direction=(vec_t - vec_s))

    def ray_from_tangent(
        self, start: Coordinates3D, direction: AbstractVector
    ) -> Ray:
        return CarthesianGeometry.Ray(start=start, direction=direction)
