# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest
from itertools import permutations

from nerte.geometry.coordinates import Coordinates
from nerte.geometry.vector import AbstractVector
from nerte.geometry.ray import Ray
from nerte.geometry.face import Face
from nerte.geometry.geometry import EuclideanGeometry
from nerte.geometry.geometry import DummyNonEuclideanGeometry


# no test for abstract class/interface Geometry


class EuclideanGeometryIntersectsTest1(unittest.TestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates(1.0, 0.0, 0.0)
        p2 = Coordinates(0.0, 1.0, 0.0)
        p3 = Coordinates(0.0, 0.0, 1.0)
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry
        self.geo = EuclideanGeometry()
        # rays pointing 'forwards' towards faces and parallel to face normal
        s0 = Coordinates(0.0, 0.0, 0.0)
        s1 = Coordinates(0.3, 0.0, 0.0)  # one third of p1
        s2 = Coordinates(0.0, 0.3, 0.0)  # one third of p2
        s3 = Coordinates(0.0, 0.0, 0.3)  # one third of p3
        ss = (s0, s1, s2, s3)
        v = AbstractVector(1.0, 1.0, 1.0)
        self.intersecting_rays = list(Ray(start=s, direction=v) for s in ss)

    def test_euclidean_intersects_1(self) -> None:
        """
        Tests if rays intersect as expected.
        Each ray points 'forwards' towards the face and is parallel to face's
        normal.
        """
        for r in self.intersecting_rays:
            for f in self.faces:
                self.assertTrue(self.geo.intersects(r, f))


class EuclideanGeometryIntersectsTest2(unittest.TestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates(1.0, 0.0, 0.0)
        p2 = Coordinates(0.0, 1.0, 0.0)
        p3 = Coordinates(0.0, 0.0, 1.0)
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry
        self.geo = EuclideanGeometry()
        # rays pointing 'backwards' and are parallel to face's normal
        s0 = Coordinates(0.0, 0.0, 0.0)
        s1 = Coordinates(0.3, 0.0, 0.0)  # one third of p1
        s2 = Coordinates(0.0, 0.3, 0.0)  # one third of p2
        s3 = Coordinates(0.0, 0.0, 0.3)  # one third of p3
        ss = (s0, s1, s2, s3)
        v = AbstractVector(1.0, 1.0, 1.0)
        self.non_intersecting_rays = list(
            Ray(start=s, direction=-v) for s in ss
        )

    def test_euclidean_intersects_2(self) -> None:
        """
        Tests if rays do not intersect as expected.
        Each ray points 'backwards' away from the face and is parallel to face's
        normal.
        """
        for r in self.non_intersecting_rays:
            for f in self.faces:
                self.assertFalse(self.geo.intersects(r, f))


class EuclideanGeometryIntersectsTest3(unittest.TestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates(1.0, 0.0, 0.0)
        p2 = Coordinates(0.0, 1.0, 0.0)
        p3 = Coordinates(0.0, 0.0, 1.0)
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry
        self.geo = EuclideanGeometry()
        # rays miss the face and are parallel to face's normal
        s1 = Coordinates(0.0, 0.6, 0.6)  # 'complement' of p1
        s2 = Coordinates(0.6, 0.0, 0.6)  # 'complement' of p2
        s3 = Coordinates(0.6, 0.6, 0.0)  # 'complement' of p3
        ss = (s1, s2, s3)
        v = AbstractVector(1.0, 1.0, 1.0)
        self.non_intersecting_rays = list(Ray(start=s, direction=v) for s in ss)

    def test_euclidean_intersects_3(self) -> None:
        """
        Tests if rays do not intersect as expected.
        Each ray misses the face and is parallel to face's normal.
        """
        for r in self.non_intersecting_rays:
            for f in self.faces:
                self.assertFalse(self.geo.intersects(r, f))


class EuclideanGeometryIntersectsTest4(unittest.TestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates(1.0, 0.0, 0.0)
        p2 = Coordinates(0.0, 1.0, 0.0)
        p3 = Coordinates(0.0, 0.0, 1.0)
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry
        self.geo = EuclideanGeometry()
        # rays completely miss the face by pointing away from it
        # and are parallel to face's normal
        s1 = Coordinates(0.0, 0.6, 0.6)  # 'complement' of p1
        s2 = Coordinates(0.6, 0.0, 0.6)  # 'complement' of p2
        s3 = Coordinates(0.6, 0.6, 0.0)  # 'complement' of p3
        ss = (s1, s2, s3)
        v = AbstractVector(1.0, 1.0, 1.0)
        self.non_intersecting_rays = list(
            Ray(start=s, direction=-v) for s in ss
        )

    def test_euclidean_intersects_4(self) -> None:
        """
        Tests if rays do not intersect as expected.
        Each ray completely misses the face and is parallel to face's normal.
        """
        for r in self.non_intersecting_rays:
            for f in self.faces:
                self.assertFalse(self.geo.intersects(r, f))


# TODO: remove with original dummy
class DummyGeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates(1.0, 0.0, 0.0)
        p2 = Coordinates(0.0, 1.0, 0.0)
        p3 = Coordinates(0.0, 0.0, 1.0)
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry (no bend == euclidean)
        self.geo = DummyNonEuclideanGeometry(
            max_steps=10, max_ray_length=10.0, bend_factor=0.0
        )
        # rays pointing 'forwards' towards faces and parallel to face normal
        s0 = Coordinates(0.0, 0.0, 0.0)
        s1 = Coordinates(0.3, 0.0, 0.0)  # one third of p1
        s2 = Coordinates(0.0, 0.3, 0.0)  # one third of p2
        s3 = Coordinates(0.0, 0.0, 0.3)  # one third of p3
        ss = (s0, s1, s2, s3)
        v = AbstractVector(1.0, 1.0, 1.0)
        self.intersecting_rays = list(Ray(start=s, direction=v) for s in ss)

    def test_dummy_geometry_intersects(self) -> None:
        """
        Tests if rays intersect as expected.
        Each ray points 'forwards' towards the face and is parallel to face's
        normal.
        """
        for r in self.intersecting_rays:
            for f in self.faces:
                self.assertTrue(self.geo.intersects(r, f))


if __name__ == "__main__":
    unittest.main()
