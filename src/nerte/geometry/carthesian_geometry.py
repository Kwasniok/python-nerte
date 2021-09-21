"""Module for representing an euclidean geometry in carthesian coordinates."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import normalized, is_zero_vector
from nerte.values.tangential_vector import TangentialVector
from nerte.values.face import Face
from nerte.values.ray_segment import RaySegment
from nerte.values.intersection_info import IntersectionInfo
from nerte.values.util.convert import coordinates_as_vector
from nerte.geometry.geometry import Geometry, intersection_ray_depth


class CarthesianGeometry(Geometry):
    """Represenation of the euclidean geometry in Carthesian coordinates."""

    class Ray(Geometry.Ray):
        """Represenation of a ray in euclidean geometry in Carthesian coordinates."""

        def __init__(self, tangential_vector: TangentialVector):
            if is_zero_vector(tangential_vector.vector):
                raise ValueError(
                    f"Cannot construct carthesian ray with zero vector"
                    f" {tangential_vector}."
                )
            vector = normalized(tangential_vector.vector)
            tangent = TangentialVector(
                point=tangential_vector.point, vector=vector
            )
            self._segment = RaySegment(
                tangential_vector=tangent, is_finite=False
            )

        def __repr__(self) -> str:
            return (
                f"CarthesianGeometry.Ray("
                f"start={self._segment.start()}"
                f", direction={self._segment.direction()})"
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
        tangent = TangentialVector(point=start, vector=(vec_t - vec_s))
        return CarthesianGeometry.Ray(tangential_vector=tangent)

    def ray_from_tangent(self, tangential_vector: TangentialVector) -> Ray:
        return CarthesianGeometry.Ray(tangential_vector=tangential_vector)
