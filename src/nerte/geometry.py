"""Module for representing a geometry."""

from abc import ABC, abstractmethod

from nerte.coordinates import Coordinates
from nerte.vector import Vector
from nerte.face import Face
from nerte.ray import Ray


class Geometry(ABC):
    """Interface of a geometry."""

    @abstractmethod
    def is_valid_coordinate(self, coordinates: Coordinates) -> bool:
        """Returns True, iff coordinates are within the valid domain."""
        # pylint: disable=W0107
        pass

    @abstractmethod
    def intersects(self, ray: Ray, face: Face) -> bool:
        """
        Returns True, iff the geodesic initiated by the ray intersects
        with the face.
        """
        # pylint: disable=W0107
        pass


# auxiliar trivial conversions
coords_to_vec = lambda c: Vector(c[0], c[1], c[2])
vec_to_coords = lambda v: Coordinates(v[0], v[1], v[2])


class EuclideanGeometry(Geometry):
    """Represenation of the euclidean geometry in Carthesian coordinates."""

    def __init__(self):
        # precision of floating point representations
        # pylint: disable=C0103,C0144
        self.𝜀 = 1e-8

    def is_valid_coordinate(self, coordinates: Coordinates) -> bool:
        return True

    def intersects(self, ray: Ray, face: Face) -> bool:
        # pylint: disable=C0103

        # (tivially) convert face coordinates to vectors
        # NOTE: is optimized for speed
        v0 = coords_to_vec(face[0])
        v1 = coords_to_vec(face[1])
        v2 = coords_to_vec(face[2])
        ## plane parameters:
        # basis vector spanning the plane
        b1 = v1 - v0
        b2 = v2 - v0
        # normal vector of plane
        n = (b1.cross(b2)).normalized()
        # level parameter (distance for plane to origin)
        l = n.dot(v0)
        # (x,y,z) in plane <=> (x,y,z) . n = l

        ## ray parameters
        s = coords_to_vec(ray.start)
        u = ray.direction
        # (x,y,z) in line <=> ∃t: s + t*u = (x,y,z)

        # intersection of line iff ∃t: (s + t*u) . n = l
        # <=> ∃t: t = a/b  for a = l - s . n and b = u . n
        # Here, b = 0 means that the line is parallel to the plane and
        # a = 0 means that s is in the plane

        ## intersection of line and plane
        # true if b≠0 or (b=0 and a=0)
        a = l - s.dot(n)
        b = u.dot(n)

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
        b1b1 = b1.dot(b1)
        b1b2 = b1.dot(b2)
        b2b2 = b2.dot(b2)
        D = b1b1 * b2b2 - b1b2 * b1b2
        b1x = x.dot(b1)
        b2x = x.dot(b2)
        f1 = (b1b1 * b2x - b1b2 * b1x) / D
        f2 = (b2b2 * b1x - b1b2 * b2x) / D

        # test if x is inside the triangle
        return f1 >= 0 and f2 >= 0 and f1 + f2 < 1


def intersects_segment(ray: Ray, face: Face) -> bool:
    """
    Returns True, iff the (straight) ray intersects with the face within
    its length.
    """

    # pylint: disable=C0103

    # (tivially) convert face coordinates to vectors
    # NOTE: is optimized for speed
    v0 = coords_to_vec(face[0])
    v1 = coords_to_vec(face[1])
    v2 = coords_to_vec(face[2])
    ## plane parameters:
    # basis vector spanning the plane
    b1 = v1 - v0
    b2 = v2 - v0
    # normal vector of plane
    n = (b1.cross(b2)).normalized()
    # level parameter (distance for plane to origin)
    l = n.dot(v0)
    # (x,y,z) in plane <=> (x,y,z) . n = l

    ## ray parameters
    s = coords_to_vec(ray.start)
    u = ray.direction
    # (x,y,z) in line <=> ∃t: s + t*u = (x,y,z)

    # intersection of line iff ∃t: (s + t*u) . n = l
    # <=> ∃t: t = a/b  for a = l - s . n and b = u . n
    # Here, b = 0 means that the line is parallel to the plane and
    # a = 0 means that s is in the plane

    ## intersection of line and plane
    # true if b≠0 or (b=0 and a=0)
    a = l - s.dot(n)
    b = u.dot(n)

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
    b1b1 = b1.dot(b1)
    b1b2 = b1.dot(b2)
    b2b2 = b2.dot(b2)
    D = b1b1 * b2b2 - b1b2 * b1b2
    b1x = x.dot(b1)
    b2x = x.dot(b2)
    f1 = (b1b1 * b2x - b1b2 * b1x) / D
    f2 = (b2b2 * b1x - b1b2 * b2x) / D

    # test if x is inside the triangle
    return f1 >= 0 and f2 >= 0 and f1 + f2 < 1


# TODO: remove?
class DummyNonEuclideanGeometry(Geometry):
    """
    Represenation of a non-euclidean geometry similar to the euclidean geometry
    but 'bends' light rays slightly.
    """

    def __init__(
        self, max_steps: int, max_ray_length: float, bend_factor: float
    ):
        # precision of floating point representations
        # pylint: disable=C0103,C0144
        self.𝜀 = 1e-8
        self.max_steps = max_steps
        self.max_ray_length = max_ray_length
        self.ray_segment_length = max_ray_length / max_steps
        self.bend_factor = bend_factor

    def is_valid_coordinate(self, coordinates: Coordinates) -> bool:
        return True

    def next_ray_segment(self, ray: Ray) -> Ray:
        """
        Returns the next segment (straight approximation) of the geodesic.
        """

        # pylint: disable=C0103

        # old
        s = ray.start
        v = ray.direction
        # new
        t = Coordinates(s[0] + v[0], s[1] + v[1], s[2] + v[2])
        # bend
        # w = v + Vector(s[0], s[1], s[2]) * (self.bend_factor * self.ray_segment_length)
        # swirl
        w = v + v.cross(Vector(s[0], s[1], s[2])) * self.bend_factor
        w = w.normalized() * self.ray_segment_length
        return Ray(start=t, direction=w)

    def intersects(self, ray: Ray, face: Face) -> bool:
        current_ray_segment = Ray(
            start=ray.start,
            direction=ray.direction.normalized() * self.ray_segment_length,
        )
        for _ in range(self.max_steps):
            if intersects_segment(current_ray_segment, face):
                return True
            current_ray_segment = self.next_ray_segment(current_ray_segment)
        return False
