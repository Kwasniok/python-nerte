"""
Module for representing a (non-euclidean) geometry where rays are propagated via
the Runge-Kutta algortihm.
"""

from typing import Optional

from abc import abstractmethod
from collections.abc import Callable

import math

from nerte.algorithm.runge_kutta import runge_kutta_4_delta
from nerte.values.coordinates import Coordinates3D
from nerte.values.face import Face
from nerte.values.tangential_vector import TangentialVector
from nerte.values.ray_segment import RaySegment
from nerte.values.ray_segment_delta import (
    RaySegmentDelta,
    ray_segment_as_delta,
    add_ray_segment_delta,
)
from nerte.values.intersection_info import IntersectionInfo, IntersectionInfos
from nerte.values.extended_intersection_info import ExtendedIntersectionInfo
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
            initial_tangent = geometry.normalized(initial_tangent)
            self._geometry = geometry
            self._initial_tangent = initial_tangent
            self._current_tangent = initial_tangent
            self._segments_and_lengths: list[
                Optional[tuple[RaySegment, float]]
            ] = [None] * geometry.max_steps()
            self._segments_cached = 0
            self._cached_ray_depth = 0.0
            self._cached_ray_left_manifold = False

        def __repr__(self) -> str:
            return (
                f"SegmentedRayGeometry.Ray("
                f"initial_tangent={self._initial_tangent})"
            )

        def initial_tangent(self) -> RaySegment:
            """Returs the initial tangent of the ray at its pointing point."""
            return self._initial_tangent

        # TODO: test cache generation (for max_steps = 0 etc.)
        def _cache_next_segment(self) -> None:
            """
            To be called to cache the next ray segment.

            :precon: self._segments_cached <= self._geometry.max_steps()
            :precon: self._cached_ray_depth <= self._geometry.max_ray_depth()
            :precon: self._geometry.is_valid_coordinate(
                         self._current_tangent.point
                     )

            :raises: RuntimeError if next segment cannot be created.
            """
            if self._cached_ray_left_manifold:
                return
            if self._segments_cached > self._geometry.max_steps():
                raise RuntimeError(
                    f"Cannot generate next ray segment for ray pointing with"
                    f" initial tangent {self._initial_tangent}."
                    f" Generating segment {self._segments_cached + 1} would"
                    f" exceed the step maximum of {self._geometry.max_steps()}."
                )
            if self._cached_ray_depth > self._geometry.max_ray_depth():
                raise RuntimeError(
                    f"Cannot generate next ray segment for ray pointing with"
                    f" initial tangent {self._initial_tangent}."
                    f" Generating segment {self._segments_cached + 1} would"
                    f" exceed the ray depth maximum of"
                    f" {self._geometry.max_ray_depth()}."
                )
            tangent = self._current_tangent
            if not self._geometry.is_valid_coordinate(tangent.start()):
                raise RuntimeError(
                    f"Cannot generate next ray segment for ray pointing with"
                    f" initial tangent {self._initial_tangent}."
                    f" Generating segment {self._segments_cached + 1} is not"
                    f" possible, since it's initial tangent={tangent} has"
                    f" invalid pointing coordinates."
                )
            geometry = self._geometry
            # calculate difference to next point/tangent on the ray
            # Note: The step size behaves like Δt where t is the
            #       parameter of the curve on which the light travels.
            # Note: The smaller the step size, the better the approximation.
            tangent_delta = runge_kutta_4_delta(
                geometry.geodesic_equation(),
                ray_segment_as_delta(tangent),
                geometry.step_size(),
            )
            segment = RaySegment(
                tangential_vector=TangentialVector(
                    point=tangent.start(),
                    vector=tangent_delta.point_delta,
                )
            )
            segment_length = geometry.length(segment)
            self._segments_and_lengths[self._segments_cached] = (
                segment,
                segment_length,
            )
            self._current_tangent = add_ray_segment_delta(
                tangent, tangent_delta
            )
            self._segments_cached += 1
            self._cached_ray_depth += segment_length
            if not geometry.is_valid_coordinate(self._current_tangent.start()):
                self._cached_ray_left_manifold = True

        def intersection_info(self, face: Face) -> IntersectionInfo:
            geometry = self._geometry

            step = 0
            total_ray_depth = 0.0
            while (
                total_ray_depth < geometry.max_ray_depth()
                and step < geometry.max_steps()
            ):
                if step == self._segments_cached:
                    self._cache_next_segment()
                sement_and_length = self._segments_and_lengths[step]

                if sement_and_length is None:
                    # ray has left the boundaries of the (local map of the)
                    # manifold
                    return IntersectionInfos.RAY_LEFT_MANIFOLD

                segment, segment_length = sement_and_length
                relative_segment_depth = intersection_ray_depth(
                    ray=segment, face=face
                )
                if relative_segment_depth < math.inf:
                    total_ray_depth += relative_segment_depth * segment_length
                    return ExtendedIntersectionInfo(
                        ray_depth=total_ray_depth, meta_data={"steps": step + 1}
                    )

                step += 1
                total_ray_depth += segment_length

            return IntersectionInfos.NO_INTERSECTION

    def __init__(
        self,
        max_ray_depth: float,
        step_size: float,
        max_steps: int,  # TODO: rename to max_segment_count
    ):
        if not max_ray_depth > 0:
            raise ValueError(
                f"Cannot create Runge-Kutta geometry."
                f" Maximum of ray length must be positive (not"
                f" {max_ray_depth})."
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
        self._max_ray_depth = max_ray_depth
        self._step_size = step_size
        self._max_steps = max_steps

    def max_ray_depth(self) -> float:
        """
        Returns the maximal depth of a ray.

        Rays may only be simulated up to this limit.
        """
        return self._max_ray_depth

    def step_size(self) -> float:
        """
        Returns the size between two sampling points for the discrete
        approximation of the ray.

        Typically, a smaller step sizes leads to improved approximations.
        """
        return self._step_size

    def max_steps(self) -> int:
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
        self, tangential_vector: TangentialVector
    ) -> "RungeKuttaGeometry.Ray":
        if not self.is_valid_coordinate(tangential_vector.point):
            raise ValueError(
                f"Cannot create Runge-Kuuta ray from tangential vector"
                f" {tangential_vector}."
                f" It is illdefined with coordinated outside the manifold."
            )
        return RungeKuttaGeometry.Ray(
            geometry=self,
            initial_tangent=RaySegment(tangential_vector=tangential_vector),
        )

    @abstractmethod
    def length(self, ray: RaySegment) -> float:
        # pylint: disable=W0107
        """
        Returns the length of the vector with respect to the tangential space.

        :raises: ValueError if ray.point are invalid coordinates
        """
        pass

    def normalized(self, ray: RaySegment) -> RaySegment:
        """
        Returns the normalized vector with respect to the tangential space.

        :raises: ValueError if ray.point are invalid coordinates
        """
        return RaySegment(
            tangential_vector=TangentialVector(
                point=ray.start(),
                vector=ray.direction() / self.length(ray),
            )
        )

    @abstractmethod
    def geodesic_equation(self) -> Callable[[RaySegmentDelta], RaySegmentDelta]:
        # pylint: disable=W0107
        """
        Returns the equation of motion for the geodesics encoded in a function
        of the trajectory configuration.
        """
        pass
