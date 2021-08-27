from abc import ABC, abstractmethod
from math import sqrt
from nerte.coordinates import Coordinates
from nerte.vector import Vector
from nerte.face import Face
from nerte.ray import Ray


class Geometry(ABC):
    @abstractmethod
    def is_valid_coordinate(self, coordinates: Coordinates) -> bool:
        pass

    @abstractmethod
    def intersects(self, ray: Ray, face: Face) -> bool:
        pass


class EuclideanGeometry(Geometry):
    def __init__(self):
        # precision
        self.𝜀 = 1e-8

    def is_valid_coordinate(self, coordinates: Coordinates) -> bool:
        return True

    def coordinates_to_vector(self, coordinates: Coordinates) -> Vector:
        return Vector(*(coordinates[i] for i in range(3)))

    def vector_to_coordinates(self, vector: Vector) -> Coordinates:
        return Coordinates(*(vector[i] for i in range(3)))

    def intersects(self, ray: Ray, face: Face) -> bool:

        # (tivially) convert face coordinates to vectors
        v0, v1, v2 = (self.coordinates_to_vector(c) for c in face)
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
        s = self.coordinates_to_vector(ray.start)
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
