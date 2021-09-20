# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144


import unittest

from typing import Optional, cast
from itertools import permutations

from nerte.values.ray_segment_unittest import RaySegmentTestCaseMixin
from nerte.geometry.geometry_unittest import GeometryTestCaseMixin

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, normalized
from nerte.values.tangential_vector import TangentialVector
from nerte.values.face import Face
from nerte.values.ray_segment import RaySegment
from nerte.geometry.carthesian_geometry import CarthesianGeometry


class CarthesianGeometryTestCaseMixin(GeometryTestCaseMixin):
    # pylint: disable=R0903
    def assertCarthRayEquiv(
        self,
        x: CarthesianGeometry.Ray,
        y: CarthesianGeometry.Ray,
        msg: Optional[str] = None,
    ) -> None:
        """
        Asserts the equivalence of two ray's.
        """

        test_case = cast(unittest.TestCase, self)
        try:
            cast(RaySegmentTestCaseMixin, self).assertRaySegmentEquiv(
                x.as_segment(), y.as_segment()
            )
        except AssertionError as ae:
            msg_full = f"Ray segment {x} is not equivalent to {y}."
            if msg is not None:
                msg_full += f" : {msg}"
            raise test_case.failureException(msg_full) from ae


class CarthesianGeometryConstructorTest(
    unittest.TestCase, GeometryTestCaseMixin
):
    def test_constructor(self) -> None:
        # pylint: disable=R0201
        """Test the constructor."""
        CarthesianGeometry()


class CarthesianGeometryIsValidCoordinateTest(
    unittest.TestCase, GeometryTestCaseMixin
):
    def setUp(self) -> None:
        self.geo = CarthesianGeometry()
        self.valid_coords = (Coordinates3D((0.0, 0.0, 0.0)),)

    def test_is_valid_coordinate(self) -> None:
        """Tests coordinate validity."""
        for coords in self.valid_coords:
            self.assertTrue(self.geo.is_valid_coordinate(coords))


class CarthesianGeometryRayFromTest(unittest.TestCase, GeometryTestCaseMixin):
    def setUp(self) -> None:
        self.geo = CarthesianGeometry()
        self.coords1 = Coordinates3D((0.0, 0.0, 0.0))
        self.coords2 = Coordinates3D((0.0, 1.0, 2.0))
        self.vector = AbstractVector((0.0, 1.0, 2.0))  # equiv to cords2
        self.tangent = TangentialVector(
            point=self.coords1, vector=normalized(self.vector)
        )
        self.init_seg = RaySegment(
            tangential_vector=self.tangent, is_finite=False
        )

    def test_ray_from_coords(self) -> None:
        """Tests ray from coordinates."""
        ray = self.geo.ray_from_coords(self.coords1, self.coords2)
        init_seg = ray.as_segment()
        self.assertRaySegmentEquiv(init_seg, self.init_seg)

    def test_ray_from_tangent(self) -> None:
        """Tests ray from tangent."""
        ray = self.geo.ray_from_tangent(self.tangent)
        init_seg = ray.as_segment()
        self.assertRaySegmentEquiv(init_seg, self.init_seg)


class CarthesianGeometryIntersectsTest1(
    unittest.TestCase, GeometryTestCaseMixin
):
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
            geo.ray_from_tangent(TangentialVector(point=s, vector=v))
            for s in ss
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


class CarthesianGeometryIntersectsTest2(
    unittest.TestCase, GeometryTestCaseMixin
):
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
            geo.ray_from_tangent(TangentialVector(point=s, vector=-v))
            for s in ss
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


class CarthesianGeometryIntersectsTest3(
    unittest.TestCase, GeometryTestCaseMixin
):
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
            geo.ray_from_tangent(TangentialVector(point=s, vector=v))
            for s in ss
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


class CarthesianGeometryIntersectsTest4(
    unittest.TestCase, GeometryTestCaseMixin
):
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
            geo.ray_from_tangent(TangentialVector(point=s, vector=-v))
            for s in ss
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
