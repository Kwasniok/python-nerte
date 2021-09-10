"""Module for representing a geometry."""

from typing import Optional
from collections.abc import Callable

from abc import ABC, abstractmethod

import math

from nerte.algorithm.runge_kutta import runge_kutta_4_delta
from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, dot, cross, length, normalized
from nerte.values.face import Face
from nerte.values.ray_segment import RaySegment
from nerte.values.ray_segment_delta import (
    RaySegmentDelta,
    ray_segment_as_delta,
    add_ray_segment_delta,
)
from nerte.values.intersection_info import IntersectionInfo
from nerte.values.util.convert import coordinates_as_vector


class Geometry(ABC):
    """Interface of a geometry."""

    @abstractmethod
    def is_valid_coordinate(self, coordinates: Coordinates3D) -> bool:
        """Returns True, iff coordinates are within the valid domain."""
        # pylint: disable=W0107
        pass

    # TODO: needs optimization: the ray is currently calculated for each
    #       use cache or allow for multiple faces at once?
    @abstractmethod
    def intersection_info(
        self, ray: RaySegment, face: Face
    ) -> IntersectionInfo:
        """
        Returns information about the intersection test of the ray and face.
        """
        # pylint: disable=W0107
        pass

    @abstractmethod
    # TODO: change to ray generator
    def initial_ray_segment_towards(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> RaySegment:
        """
        Returns the initial ray segment from a ray, which starts at the given
        position and passes the target.

        :raises: ValueError if no valid ray could be constructed
        """
        # pylint: disable=W0107
        pass


def _in_triangle(
    b1: AbstractVector, b2: AbstractVector, x: AbstractVector
) -> bool:
    # pylint: disable=C0103
    """
    Returns True, if x denotes a point within the triangle spanned by b1 and b2.
    ASSERTION: It was previously checked that x lies in the plane spanned by
               b1 and b2. Otherwise this test is meaningless.
    """
    # solve:
    #   x = f1 * v1 + f2 * v2
    # <=>
    #   v1 . x = f1 * v1 . v1 + f2 * v1 . v2
    #   v2 . x = f1 * v2 . v1 + f2 * v2 . v2
    # <=>
    #   B = A * F where
    #       B = ⎛v1 . x⎞
    #           ⎝v2 . x⎠
    #       A = ⎛v1 . v1    v1 . v2⎞
    #           ⎝v2 . v1    v2 . v2⎠
    #       F = ⎛f1⎞
    #           ⎝f2⎠
    b1b1 = dot(b1, b1)
    b1b2 = dot(b1, b2)
    b2b2 = dot(b2, b2)
    D = b1b1 * b2b2 - b1b2 * b1b2
    b1x = dot(x, b1)
    b2x = dot(x, b2)
    f1 = (b1b1 * b2x - b1b2 * b1x) / D
    f2 = (b2b2 * b1x - b1b2 * b2x) / D

    # test if x is inside the triangle
    return f1 >= 0 and f2 >= 0 and f1 + f2 <= 1


def intersection_ray_depth(
    ray: RaySegment, is_ray_segment: bool, face: Face
) -> float:
    """
    Returns relative ray depth of intersection point or math.inf if no
    intersection occurred.

    Note: If the returned value t is finite, the intersection occurred at
          x = ray.start + ray.direction * t
    """
    # pylint: disable=C0103

    # (tivially) convert face coordinates to vectors
    v0 = coordinates_as_vector(face[0])
    v1 = coordinates_as_vector(face[1])
    v2 = coordinates_as_vector(face[2])
    ## plane parameters:
    # basis vector spanning the plane
    b1 = v1 - v0
    b2 = v2 - v0
    # normal vector of plane
    n = normalized(cross(b1, b2))
    # level parameter (distance for plane to origin)
    l = dot(n, v0)
    # (x,y,z) in plane <=> (x,y,z) . n = l

    ## ray parameters
    s = coordinates_as_vector(ray.start)
    u = ray.direction
    # (x,y,z) in line <=> ∃t: s + t*u = (x,y,z)

    # intersection of line iff ∃t: (s + t*u) . n = l
    # <=> ∃t: t = a/b  for a = l - s . n and b = u . n
    # Here, b = 0 means that the line is parallel to the plane and
    # a = 0 means that s is in the plane

    ## intersection of line and plane
    # true if b≠0 or (b=0 and a=0)
    a = l - dot(s, n)
    b = dot(u, n)

    if b == 0:
        # ray is parallel to plane
        if a == 0:
            # ray starts inside plane
            return 0.0  # this value somewhat arbitrary
        # ray starts outside of plane
        return math.inf  # no intersection possible

    t = a / b

    if t < 0:
        # intersection is before ray segment started
        return math.inf
    if is_ray_segment and t > 1:
        # intersection after ray segment ended
        return math.inf

    # x = intersection point with respect to the triangles origin
    # return if x lies in the triangle spanned by b1 and b2
    if _in_triangle(b1, b2, (s + u * t) - v0):
        return t
    return math.inf


class CarthesianGeometry(Geometry):
    """Represenation of the euclidean geometry in Carthesian coordinates."""

    def __init__(self) -> None:
        pass

    def is_valid_coordinate(self, coordinates: Coordinates3D) -> bool:
        return True

    def intersection_info(
        self, ray: RaySegment, face: Face
    ) -> IntersectionInfo:
        ray_depth = intersection_ray_depth(
            ray=ray, is_ray_segment=False, face=face
        ) * length(ray.direction)
        return IntersectionInfo(ray_depth=ray_depth)

    def initial_ray_segment_towards(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> RaySegment:
        vec_s = coordinates_as_vector(start)
        vec_t = coordinates_as_vector(target)
        return RaySegment(start=start, direction=(vec_t - vec_s))


class SegmentedRayGeometry(Geometry):
    """
    Represenation of a non-euclidean geometry where rays are bend in space
    ans approximated with staright short ray segments.
    """

    def __init__(self, max_steps: int, max_ray_length: float):

        if not max_steps > 0:
            raise ValueError(
                "Cannot create segmented ray geometry. max_steps must be strictly"
                + f" positive (given value is {max_steps})"
            )
        if not 0.0 < max_ray_length < math.inf:
            raise ValueError(
                "Cannot create segmented ray geometry. max_ray_length must be strictly"
                + f" positive and finite (given value is {max_ray_length})"
            )

        self.max_steps = max_steps
        self.max_ray_length = max_ray_length
        self._ray_segment_length = max_ray_length / max_steps

    def ray_segment_length(self) -> float:
        """Returns the length of each ray segment."""
        return self._ray_segment_length

    @abstractmethod
    def next_ray_segment(self, ray: RaySegment) -> Optional[RaySegment]:
        # pylint: disable=W0107
        """
        Returns the next ray segment (straight approximation of the geodesic
        segment) if it exists.

        NOTE: A ray might hit the boundary of the manifold representing the
              geometry. If this happens further extending the ray might be
              infeasable.
        """
        pass

    @abstractmethod
    def normalize_initial_ray_segment(self, ray: RaySegment) -> RaySegment:
        # pylint: disable=W0107
        """
        Returns the first ray segment (straight approximation of the geodesic
        segment) based on a given ray.
        """
        pass

    def intersection_info(
        self, ray: RaySegment, face: Face
    ) -> IntersectionInfo:
        current_ray_segment = self.normalize_initial_ray_segment(ray)
        for step in range(self.max_steps):
            if not self.is_valid_coordinate(current_ray_segment.start):
                raise ValueError(
                    f"Cannot test for intersection for ray={ray}"
                    f" and face={face}."
                    f"At step={step} a ray segment={current_ray_segment} was"
                    f" created which has invalid starting coordinates."
                )

            relative_segment_ray_depth = intersection_ray_depth(
                ray=current_ray_segment, is_ray_segment=True, face=face
            )
            if relative_segment_ray_depth < math.inf:
                total_ray_depth = (
                    step + relative_segment_ray_depth
                ) * self.ray_segment_length()
                return IntersectionInfo(ray_depth=total_ray_depth)
            next_ray_segment = self.next_ray_segment(current_ray_segment)
            if next_ray_segment is not None:
                current_ray_segment = next_ray_segment
            else:
                return IntersectionInfo(
                    miss_reasons=set(
                        (IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD,)
                    )
                )
        return IntersectionInfo(ray_depth=math.inf)


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
                ray=ray_segment, is_ray_segment=True, face=face
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
