import unittest
from itertools import permutations

from nerte.coordinates import Coordinates
from nerte.vector import Vector
from nerte.ray import Ray
from nerte.face import Face
from nerte.geometry import EuclideanGeometry

# no test for abstract class/interface Geometry


# face coordinates
p1 = Coordinates(1.0, 0.0, 0.0)
p2 = Coordinates(0.0, 1.0, 0.0)
p3 = Coordinates(0.0, 0.0, 1.0)

# same face with all permuations of coordinates
# NOTE: Results are invariant under coordinate permutation!
faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))


class EuclideanGeometryTest(unittest.TestCase):
    def test_intersects_1(self):
        # rays starting 'close' to the face
        # all orthogonal to the face#
        # all intersect

        geo = EuclideanGeometry()

        # ray starting points
        s0 = Coordinates(0.0, 0.0, 0.0)
        s1 = Coordinates(0.3, 0.0, 0.0)  # one third of p1
        s2 = Coordinates(0.0, 0.3, 0.0)  # one third of p2
        s3 = Coordinates(0.0, 0.0, 0.3)  # one third of p3
        ss = (s0, s1, s2, s3)
        # ray directions
        v = Vector(1.0, 1.0, 1.0)  # parallel to face normal
        # rays pointing 'forwards'
        intersecting_rays = list(Ray(start=s, direction=v) for s in ss)

        # intersection tests
        for r in intersecting_rays:
            for f in faces:
                self.assertTrue(geo.intersects(r, f))

    def test_intersects_2(self):
        # rays starting 'close' to the face
        # all orthogonal to the face#
        # none intersect

        geo = EuclideanGeometry()

        # ray starting points
        s0 = Coordinates(0.0, 0.0, 0.0)
        s1 = Coordinates(0.3, 0.0, 0.0)  # one third of p1
        s2 = Coordinates(0.0, 0.3, 0.0)  # one third of p2
        s3 = Coordinates(0.0, 0.0, 0.3)  # one third of p3
        ss = (s0, s1, s2, s3)
        # ray directions
        v = Vector(1.0, 1.0, 1.0)  # parallel to face normal
        # rays pointing 'backwards'
        non_intersecting_rays = list(Ray(start=s, direction=-v) for s in ss)

        # intersection tests
        for r in non_intersecting_rays:
            for f in faces:
                self.assertFalse(geo.intersects(r, f))

    def test_intersects_3(self):
        # rays starting 'near' the face
        # all orthogonal to the face
        # none intersect

        geo = EuclideanGeometry()

        # rays starting points
        s1 = Coordinates(0.0, 0.6, 0.6)  # 'complement' of p1
        s2 = Coordinates(0.6, 0.0, 0.6)  # 'complement' of p2
        s3 = Coordinates(0.6, 0.6, 0.0)  # 'complement' of p3
        ss = (s1, s2, s3)
        # ray directions
        v = Vector(1.0, 1.0, 1.0)  # parallel to face normal
        # rays pointing 'forwards'
        non_intersecting_rays = list(Ray(start=s, direction=v) for s in ss)

        # intersection tests
        for r in non_intersecting_rays:
            for f in faces:
                self.assertFalse(geo.intersects(r, f))

    def test_intersects_4(self):
        # rays starting 'near' the face
        # all orthogonal to the face
        # none intersect

        geo = EuclideanGeometry()

        # rays starting points
        s1 = Coordinates(0.0, 0.6, 0.6)  # 'complement' of p1
        s2 = Coordinates(0.6, 0.0, 0.6)  # 'complement' of p2
        s3 = Coordinates(0.6, 0.6, 0.0)  # 'complement' of p3
        ss = (s1, s2, s3)
        # ray directions
        v = Vector(1.0, 1.0, 1.0)  # parallel to face normal
        # rays point 'backwards'
        non_intersecting_rays = list(Ray(start=s, direction=-v) for s in ss)

        # intersection tests
        for r in non_intersecting_rays:
            for f in faces:
                self.assertFalse(geo.intersects(r, f))


if __name__ == "__main__":
    unittest.main()
