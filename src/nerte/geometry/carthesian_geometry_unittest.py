# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144


import unittest

from itertools import permutations

from nerte.geometry.geometry_unittest import GeometryTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.ray_segment import RaySegment
from nerte.values.face import Face
from nerte.geometry.carthesian_geometry import CarthesianGeometry


class CarthesianGeometryIntersectsTest1(GeometryTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry
        geo = CarthesianGeometry()
        # rays pointing 'forwards' towards faces and parallel to face normal
        s0 = Coordinates3D((0.0, 0.0, 0.0))
        s1 = Coordinates3D((0.3, 0.0, 0.0))  # one third of p1
        s2 = Coordinates3D((0.0, 0.3, 0.0))  # one third of p2
        s3 = Coordinates3D((0.0, 0.0, 0.3))  # one third of p3
        ss = (s0, s1, s2, s3)
        v = AbstractVector((1.0, 1.0, 1.0))
        self.intersecting_rays = list(
            geo.ray_from_tangent(start=s, direction=v) for s in ss
        )

    def test_euclidean_intersects_1(self) -> None:
        """
        Tests if rays intersect as expected.
        Each ray points 'forwards' towards the face and is parallel to face's
        normal.
        """
        for r in self.intersecting_rays:
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.hits())


class CarthesianGeometryIntersectsTest2(GeometryTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry
        geo = CarthesianGeometry()
        # rays pointing 'backwards' and are parallel to face's normal
        s0 = Coordinates3D((0.0, 0.0, 0.0))
        s1 = Coordinates3D((0.3, 0.0, 0.0))  # one third of p1
        s2 = Coordinates3D((0.0, 0.3, 0.0))  # one third of p2
        s3 = Coordinates3D((0.0, 0.0, 0.3))  # one third of p3
        ss = (s0, s1, s2, s3)
        v = AbstractVector((1.0, 1.0, 1.0))
        self.non_intersecting_rays = list(
            geo.ray_from_tangent(start=s, direction=-v) for s in ss
        )

    def test_euclidean_intersects_2(self) -> None:
        """
        Tests if rays do not intersect as expected.
        Each ray points 'backwards' away from the face and is parallel to face's
        normal.
        """
        for r in self.non_intersecting_rays:
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.misses())


class CarthesianGeometryIntersectsTest3(GeometryTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry
        geo = CarthesianGeometry()
        # rays miss the face and are parallel to face's normal
        s1 = Coordinates3D((0.0, 0.6, 0.6))  # 'complement' of p1
        s2 = Coordinates3D((0.6, 0.0, 0.6))  # 'complement' of p2
        s3 = Coordinates3D((0.6, 0.6, 0.0))  # 'complement' of p3
        ss = (s1, s2, s3)
        v = AbstractVector((1.0, 1.0, 1.0))
        self.non_intersecting_rays = list(
            geo.ray_from_tangent(start=s, direction=v) for s in ss
        )

    def test_euclidean_intersects_3(self) -> None:
        """
        Tests if rays do not intersect as expected.
        Each ray misses the face and is parallel to face's normal.
        """
        for r in self.non_intersecting_rays:
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.misses())


class CarthesianGeometryIntersectsTest4(GeometryTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry
        geo = CarthesianGeometry()
        # rays completely miss the face by pointing away from it
        # and are parallel to face's normal
        s1 = Coordinates3D((0.0, 0.6, 0.6))  # 'complement' of p1
        s2 = Coordinates3D((0.6, 0.0, 0.6))  # 'complement' of p2
        s3 = Coordinates3D((0.6, 0.6, 0.0))  # 'complement' of p3
        ss = (s1, s2, s3)
        v = AbstractVector((1.0, 1.0, 1.0))
        self.non_intersecting_rays = list(
            geo.ray_from_tangent(start=s, direction=-v) for s in ss
        )

    def test_euclidean_intersects_4(self) -> None:
        """
        Tests if rays do not intersect as expected.
        Each ray completely misses the face and is parallel to face's normal.
        """
        for r in self.non_intersecting_rays:
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.misses())


if __name__ == "__main__":
    unittest.main()
