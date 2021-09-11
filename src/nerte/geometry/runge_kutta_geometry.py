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
from nerte.values.linalg import AbstractVector
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
    of the equation of geodesics using a Runge-Kutta method.

    Note: Rays may be represented in any three dimensional coordinate system and
    this class is designed for rays which travel on non-linear geodesics - i.e.
    when the underlying geometry is curved/non-euclidean.

    Note: Rule of thumb: Decrease the step size to increase the accuracy of
    the rays approximation. This will also increase resource demands
    significantly!

    Note: Each ray is a curve 𝛤 (trajectory) in space parameterized by a
    curve parameter 𝜆 (time). The Runge-Kutta algorithm yields a discrete
    approximation of f at sampling points 𝜆_1, 𝜆_2, ... separated by the
    step size 𝛥𝜆 (discrete change in time) each. The discrete points are then
    connected via linear interpolation. The latter is a good approximation,
    when the space is approximately flat in the region traversed.
    The length of the initial tangent vector (velocity) times the step
    size determines the granularity of the approximation (distance covered in
    each time interval).
    The geometry's equation of geodesics determines the change in the tangent
    vector with 𝜆 (acceleration). Picking a appropriate step size can be
    non-trivial.
    """

    # TODO: tests
    class Ray(Geometry.Ray):
        """
        Representation of a ray in a Runge-Kutta geometry.

        Each ray is a discretized approximation of a smooth curve as a polygonal
        chain. The approximation consists of finitely many sampling points.
        Each point has a local tangent which approximates the actual tangent and
        may differ from the difference vector to the next point. Therefore, ray
        tangents and segments must be distinguished!
        """

        def __init__(
            self, geometry: "RungeKuttaGeometry", initial_tangent: RaySegment
        ) -> None:
            self._geometry = geometry
            self._initial_tangent = geometry.normalized(initial_tangent)

        def initial_tangent(self) -> RaySegment:
            """Returs the initial tangent of the ray at its starting point."""
            return self._initial_tangent

        def intersection_info(self, face: Face) -> IntersectionInfo:
            geometry = self._geometry
            steps = 0
            total_ray_depth = 0.0
            tangent = self._initial_tangent

            while (
                total_ray_depth < geometry.max_ray_length()
                and steps < geometry.max_steps()
            ):

                if not geometry.is_valid_coordinate(tangent.start):
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
                tangent_delta = runge_kutta_4_delta(
                    geometry.geodesic_equation(),
                    ray_segment_as_delta(tangent),
                    geometry.step_size(),
                )

                # representation of the change of the ray's position as a ray
                # segment
                segment = RaySegment(
                    start=tangent.start, direction=tangent_delta.coords_delta
                )

                relative_segment_depth = intersection_ray_depth(
                    ray=segment, face=face
                )
                if relative_segment_depth < math.inf:
                    total_ray_depth += relative_segment_depth * geometry.length(
                        segment
                    )
                    return IntersectionInfo(ray_depth=total_ray_depth)

                steps += 1
                total_ray_depth += geometry.length(segment)
                tangent = add_ray_segment_delta(tangent, tangent_delta)

            return IntersectionInfo(ray_depth=math.inf)

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

    # TODO: rename to max_ray_depth
    def max_ray_length(self) -> float:
        """
        Returns the maximal depth of a ray.

        Rays may only be simulated up to this limit.
        """
        return self._max_ray_length

    def step_size(self) -> float:
        """
        Returns the size between two sampling points for the discrete
        approximation of the ray.

        Typically, a smaller step sizes leads to improved approximations.
        """
        return self._step_size

    def max_steps(self) -> float:
        """
        Returns the limit of approximation steps.

        No simulation of a ray will have more then this amount of sampling points.
        """
        return self._max_steps

    @abstractmethod
    def ray_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> "RungeKuttaGeometry.Ray":
        pass

    def ray_from_tangent(
        self, start: Coordinates3D, direction: AbstractVector
    ) -> "RungeKuttaGeometry.Ray":
        if not self.is_valid_coordinate(start):
            raise ValueError(
                f"Cannot create ray from tangent."
                f" Start coordinates {start} are invalid."
            )
        return RungeKuttaGeometry.Ray(
            geometry=self,
            initial_tangent=RaySegment(start=start, direction=direction),
        )

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