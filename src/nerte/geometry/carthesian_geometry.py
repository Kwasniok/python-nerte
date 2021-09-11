"""Module for representing an euclidean geometry in carthesian coordinates."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import length
from nerte.values.face import Face
from nerte.values.ray_segment import RaySegment
from nerte.values.intersection_info import IntersectionInfo
from nerte.values.util.convert import coordinates_as_vector
from nerte.geometry.geometry import Geometry, intersection_ray_depth


class CarthesianGeometry(Geometry):
    """Represenation of the euclidean geometry in Carthesian coordinates."""

    def __init__(self) -> None:
        pass

    def is_valid_coordinate(self, coordinates: Coordinates3D) -> bool:
        return True

    def intersection_info(
        self, ray: RaySegment, face: Face
    ) -> IntersectionInfo:
        ray_depth = intersection_ray_depth(ray=ray, face=face) * length(
            ray.direction
        )
        return IntersectionInfo(ray_depth=ray_depth)

    def initial_ray_segment_towards(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> RaySegment:
        vec_s = coordinates_as_vector(start)
        vec_t = coordinates_as_vector(target)
        return RaySegment(
            start=start, direction=(vec_t - vec_s), is_finite=False
        )
