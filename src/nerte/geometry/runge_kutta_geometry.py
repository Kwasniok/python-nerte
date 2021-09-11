"""
Module for representing a (non-euclidean) geometry where rays are propagated via
the Runge-Kutta algortihm.
"""

from abc import abstractmethod
from collections.abc import Callable

import math

from nerte.algorithm.runge_kutta import runge_kutta_4_delta
from nerte.values.coordinates import Coordinates3D
from nerte.values.face import Face
from nerte.values.ray_segment import RaySegment
from nerte.values.ray_segment_delta import (
    RaySegmentDelta,
    ray_segment_as_delta,
    add_ray_segment_delta,
)
from nerte.values.intersection_info import IntersectionInfo
from nerte.geometry.geometry import Geometry, intersection_ray_depth


class RungeKuttaGeometry(Geometry):
    """
    Represenation of a geometry where rays are calculated based on the knowledge
    of the equation of geodesics using a Runge-Kutta method to solve it.

    Note: Rays may be represented in any three dimensional coordinate system and
    this class is designed for rays which travel on non-liner lines - i.e. when
    the underlying geometry is curved or even non-euclidean.
    """

    def __init__(
        self,
        max_ray_length: float,
        step_size: float,
        max_steps: int,
    ):
        if not max_ray_length > 0:
            raise ValueError(
                f"Cannot create Runge-Kutta geometry."
                f" Maximum of ray length must be positive (not"
                f" {max_ray_length})."
            )
        if not 0 < step_size < math.inf:
            raise ValueError(
                f"Cannot create Runge-Kutta geometry."
                f" Step size must be positive and finite (not"
                f" {step_size})."
            )

        if not max_steps > 0:
            raise ValueError(
                f"Cannot create Runge-Kutta geometry."
                f" Maximum of steps must be positive (not {max_steps})."
            )
        self._max_ray_length = max_ray_length
        self._step_size = step_size
        self._max_steps = max_steps

    @abstractmethod
    def length(self, ray: RaySegment) -> float:
        # pylint: disable=W0107
        """
        Returns the length of the vector with respect to the tangential space.

        :raises: ValueError if ray.start are invalid coordinates
        """
        pass

    def normalized(self, ray: RaySegment) -> RaySegment:
        """
        Returns the normalized vector with respect to the tangential space.

        :raises: ValueError if ray.start are invalid coordinates
        """
        return RaySegment(ray.start, ray.direction / self.length(ray))

    @abstractmethod
    def geodesic_equation(self) -> Callable[[RaySegmentDelta], RaySegmentDelta]:
        # pylint: disable=W0107
        """
        Returns the equation of motion for the geodesics encoded in a function
        of the trajectory configuration.
        """
        pass

    def intersection_info(
        self, ray: RaySegment, face: Face
    ) -> IntersectionInfo:
        steps = 0
        total_ray_depth = 0.0
        ray = self.normalized(ray)

        while (
            total_ray_depth < self._max_ray_length and steps < self._max_steps
        ):

            if not self.is_valid_coordinate(ray.start):
                # ray has left the boundaries of the (local map of the)
                # manifold
                return IntersectionInfo(
                    miss_reasons=set(
                        (IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD,)
                    )
                )

            # change in ray's configuration for a (small) step size
            # Note: The step size behaves like Δt where t is the
            #       parameter of the curve on which the light travels.
            # Note: The smaller the step size, the better the approximation.
            # TODO: check if Runge-Kutta-Nystrom is more suitable/efficient
            ray_delta = runge_kutta_4_delta(
                self.geodesic_equation(),
                ray_segment_as_delta(ray),
                self._step_size,
            )

            # representation of the change of the ray's position as a ray
            # segment
            ray_segment = RaySegment(
                start=ray.start, direction=ray_delta.coords_delta
            )

            relative_ray_segment_depth = intersection_ray_depth(
                ray=ray_segment, face=face
            )
            if relative_ray_segment_depth < math.inf:
                total_ray_depth += relative_ray_segment_depth * self.length(
                    ray_segment
                )
                return IntersectionInfo(ray_depth=total_ray_depth)

            steps += 1
            total_ray_depth += self.length(ray_segment)
            ray = add_ray_segment_delta(ray, ray_delta)

        return IntersectionInfo(ray_depth=math.inf)

    @abstractmethod
    def initial_ray_segment_towards(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> RaySegment:
        pass
