"""Bse module for representing a geometry."""

from abc import ABC, abstractmethod

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import (
    AbstractVector,
    dot,
    cross,
    normalized,
    is_zero_vector,
)
from nerte.values.tangential_vector import TangentialVector
from nerte.values.util.convert import coordinates_as_vector
from nerte.values.ray_segment import RaySegment
from nerte.values.face import Face
from nerte.values.intersection_info import IntersectionInfo


class Geometry(ABC):
    """Interface of a geometry."""

    class Ray(ABC):
        # pylint: disable=R0903
        """Interface for rays embedded in a geometry."""

        @abstractmethod
        def intersection_info(self, face: Face) -> IntersectionInfo:
            """
            Returns information about the intersection test of a ray
            with a face.
            """
            # pylint: disable=W0107
            pass

    @abstractmethod
    def are_valid_coords(self, coords: Coordinates3D) -> bool:
        """Returns True, iff coordinates are within the valid domain."""
        # pylint: disable=W0107
        pass

    @abstractmethod
    def ray_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> Ray:
        """
        Returns the ray, which starts at the given position and passes through
        the target eventually.

        :raises: ValueError if no valid ray could be constructed
        """
        # pylint: disable=W0107
        pass

    @abstractmethod
    def ray_from_tangent(self, initial_tangent: TangentialVector) -> Ray:
        """
        Returns the ray, whith the given starting point and direction.

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


def intersection_ray_depth(ray: RaySegment, face: Face) -> float:
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
    s = coordinates_as_vector(ray.start())
    u = ray.direction()
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
    if ray.is_finite and t > 1:
        # intersection after ray segment ended
        return math.inf

    # x = intersection point with respect to the triangles origin
    # return if x lies in the triangle spanned by b1 and b2
    if _in_triangle(b1, b2, (s + u * t) - v0):
        return t
    return math.inf


class StandardGeometry(Geometry):
    """
    Represenation of the euclidean geometry in Cartesian coordinates.

    Note: Can be used for mocking as well.
    """

    class Ray(Geometry.Ray):
        """Represenation of a ray in euclidean geometry in Cartesian coordinates."""

        def __init__(self, tangential_vector: TangentialVector):
            if is_zero_vector(tangential_vector.vector):
                raise ValueError(
                    f"Cannot construct cartesian ray with zero vector"
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
                f"CartesianGeometry.Ray("
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

    def are_valid_coords(self, coords: Coordinates3D) -> bool:
        return (
            math.isfinite(coords[0])
            and math.isfinite(coords[1])
            and math.isfinite(coords[2])
        )

    def ray_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> Ray:
        if not self.are_valid_coords(start) or not self.are_valid_coords(
            target
        ):
            raise ValueError(
                f"Cannot create standard ray from coordinates start={start}"
                f" and target={target}. Coordinates must consist of finite values."
            )
        vec_s = coordinates_as_vector(start)
        vec_t = coordinates_as_vector(target)
        tangent = TangentialVector(point=start, vector=(vec_t - vec_s))
        return StandardGeometry.Ray(tangential_vector=tangent)

    def ray_from_tangent(self, initial_tangent: TangentialVector) -> Ray:
        return StandardGeometry.Ray(tangential_vector=initial_tangent)
