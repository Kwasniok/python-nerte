"""Module for representing a geometry."""

from typing import Optional

from abc import ABC, abstractmethod

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, dot, cross, normalized
from nerte.values.face import Face
from nerte.values.ray import Ray
from nerte.values.util.convert import coordinates_as_vector


# TODO: make geometry a manifold as well?
# TODO: coordinates' validity must be checked
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
    def intersects(self, ray: Ray, face: Face) -> bool:
        """
        Returns True, iff the geodesic initiated by the ray intersects
        with the face.
        """
        # pylint: disable=W0107
        pass

    @abstractmethod
    def ray_towards(self, start: Coordinates3D, target: Coordinates3D) -> Ray:
        """
        Returns a ray from the given starting position pointing towards the
        given target.
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
    return f1 >= 0 and f2 >= 0 and f1 + f2 < 1


class CarthesianGeometry(Geometry):
    """Represenation of the euclidean geometry in Carthesian coordinates."""

    def __init__(self) -> None:
        # precision of floating point representations
        # pylint: disable=C0103,C0144
        self.𝜀: float = 1e-8

    def is_valid_coordinate(self, coordinates: Coordinates3D) -> bool:
        return True

    # TOD: unify with intersects_segment
    def intersects(self, ray: Ray, face: Face) -> bool:
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
            if a != 0:
                # no intersection possible
                return False
            # ray starts in plane
            return True  # this is arbitrary

        t = a / b

        if t < 0:
            # intersection is before ray start
            return False
        # ray points towards plane

        # intersection point with respect to the triangles origin
        x = (s + u * t) - v0

        # return if x lies in the triangle spanned by b1 and b2
        return _in_triangle(b1, b2, x)

    def ray_towards(self, start: Coordinates3D, target: Coordinates3D) -> Ray:
        vec_s = coordinates_as_vector(start)
        vec_t = coordinates_as_vector(target)
        return Ray(start=start, direction=(vec_t - vec_s))


def intersects_segment(ray: Ray, face: Face) -> bool:
    """
    Returns True, iff the (straight) ray intersects with the face within
    its length.
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
        if a != 0:
            # no intersection possible
            return False
        # ray starts in plane
        return True  # this is arbitrary

    t = a / b

    if t < 0:
        # intersection is before ray segment started
        return False

    # NOTE: THIS CONDITION IS UNIQUE FOR SEGMENTS!
    if t > 1:
        # intersection after ray segment ended
        return False

    # intersection point with respect to the triangles origin
    x = (s + u * t) - v0

    # return if x lies in the triangle spanned by b1 and b2
    return _in_triangle(b1, b2, x)


class SegmentedRayGeometry(Geometry):
    """
    Represenation of a non-euclidean geometry where rays are bend in space
    ans approximated with staright short ray segments.
    """

    def __init__(self, max_steps: int, max_ray_length: float):
        # precision of floating point representations
        # pylint: disable=C0103,C0144
        self.𝜀 = 1e-8
        self.max_steps = max_steps
        self.max_ray_length = max_ray_length

    @abstractmethod
    def next_ray_segment(self, ray: Ray) -> Optional[Ray]:
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
    def normalize_initial_ray(self, ray: Ray) -> Ray:
        # pylint: disable=W0107
        """
        Returns the first ray segment (straight approximation of the geodesic
        segment) based on a given ray.
        """
        pass

    def intersects(self, ray: Ray, face: Face) -> bool:
        current_ray_segment = self.normalize_initial_ray(ray)
        for _ in range(self.max_steps):
            if intersects_segment(current_ray_segment, face):
                return True
            next_ray_segment = self.next_ray_segment(current_ray_segment)
            if next_ray_segment is not None:
                current_ray_segment = next_ray_segment
            else:
                return False
        return False
