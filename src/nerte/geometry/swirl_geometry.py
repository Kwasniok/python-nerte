"""Module for representing a dummy geometry."""
# TODO: remove this module? This geometry does not converge properly.
import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.util.convert import coordinates_as_vector
from nerte.values.ray_segment import RaySegment
from nerte.values.linalg import AbstractVector, cross, normalized
from nerte.geometry.segmented_ray_geometry import SegmentedRayGeometry


class SwirlGeometry(SegmentedRayGeometry):
    """
    Represenation of a non-euclidean geometry similar to the euclidean geometry
    but 'bends' light rays slightly.
    """

    def __init__(
        self, max_steps: int, max_ray_length: float, bend_factor: float
    ):
        SegmentedRayGeometry.__init__(self, max_steps, max_ray_length)

        if math.isinf(bend_factor) or math.isnan(bend_factor):
            raise ValueError(
                "Cannot create swirl geometry."
                + f" bend_factor must be finite (given value is {bend_factor})"
            )

        self.bend_factor = bend_factor

    def is_valid_coordinate(self, coordinates: Coordinates3D) -> bool:
        return True

    def initial_ray_segment_towards(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> RaySegment:
        # TODO: This method is crude and incorrect.
        vec_s = coordinates_as_vector(start)
        vec_t = coordinates_as_vector(target)
        return RaySegment(start=start, direction=(vec_t - vec_s))

    def next_ray_segment(self, ray: RaySegment) -> RaySegment:
        # pylint: disable=C0103

        # old segment
        s_old = ray.start
        d_old = ray.direction
        # new segment
        # advance starting point
        s_new = Coordinates3D(
            (s_old[0] + d_old[0], s_old[1] + d_old[1], s_old[2] + d_old[2])
        )
        # swirl: rotate direction slightly
        # NOTE: d_new is never a zero vector
        # TODO This method does not converge properly for increased max_step
        #      and ray_segment_length.
        d_new = (
            d_old
            + cross(d_old, AbstractVector((s_old[0], s_old[1], s_old[2])))
            * self.bend_factor
        )
        # ensure ray segment length
        # NOTE: No exception handling, since d_new is never a zero vector
        d_new = normalized(d_new) * self.ray_segment_length()
        return RaySegment(start=s_new, direction=d_new)

    def normalize_initial_ray_segment(self, ray: RaySegment) -> RaySegment:
        return RaySegment(
            start=ray.start,
            direction=normalized(ray.direction) * self.ray_segment_length(),
        )
