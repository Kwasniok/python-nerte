# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest


from itertools import permutations
import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, normalized
from nerte.values.tangential_vector import TangentialVector
from nerte.values.ray_segment import RaySegment
from nerte.values.ray_segment_unittest import ray_segment_equiv
from nerte.values.face import Face
from nerte.geometry.base import intersection_ray_depth, StandardGeometry


class IntersectionRayDepthTest(BaseTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # rays
        s10 = Coordinates3D((0.0, 0.0, 0.0))
        s11 = Coordinates3D((1 / 3, 0.0, 0.0))  # one third of p1
        s12 = Coordinates3D((0.0, 1 / 3, 0.0))  # one third of p2
        s13 = Coordinates3D((0.0, 0.0, 1 / 3))  # one third of p3
        ss1 = (s10, s11, s12, s13)
        # NOTE: distance vector
        v = AbstractVector((1.0, 1.0, 1.0))
        # rays pointing 'forwards'
        # towards the face and parallel to face normal
        self.intersecting_rays = [
            RaySegment(
                tangential_vector=TangentialVector(point=s, vector=v * 0.1),
                is_finite=False,
            )
            for s in ss1
        ]
        self.ray_depths = [10 / 3, 20 / 9, 20 / 9, 20 / 9]
        self.intersecting_ray_segments = [
            RaySegment(
                tangential_vector=TangentialVector(point=s, vector=v * 1.0)
            )
            for s in ss1
        ]
        self.ray_segment_depths = [1 / 3, 2 / 9, 2 / 9, 2 / 9]
        self.non_intersecting_ray_segments = [
            RaySegment(
                tangential_vector=TangentialVector(point=s, vector=v * 0.1)
            )
            for s in ss1
        ]
        # rays pointing 'backwards'
        # away from the face and parallel to face normal
        self.non_intersecting_rays = [
            RaySegment(
                tangential_vector=TangentialVector(point=s, vector=-v),
                is_finite=False,
            )
            for s in ss1
        ]
        self.non_intersecting_ray_segments += [
            RaySegment(tangential_vector=TangentialVector(point=s, vector=-v))
            for s in ss1
        ]
        s21 = Coordinates3D((0.0, 0.6, 0.6))  # 'complement' of p1
        s22 = Coordinates3D((0.6, 0.0, 0.6))  # 'complement' of p2
        s23 = Coordinates3D((0.6, 0.6, 0.0))  # 'complement' of p3
        ss2 = (s21, s22, s23)
        # rays parallel to face normal but pointing 'outside' the face
        self.non_intersecting_rays += [
            RaySegment(
                tangential_vector=TangentialVector(point=s, vector=v),
                is_finite=False,
            )
            for s in ss2
        ]
        self.non_intersecting_rays += [
            RaySegment(
                tangential_vector=TangentialVector(point=s, vector=-v),
                is_finite=False,
            )
            for s in ss2
        ]
        self.non_intersecting_ray_segments += [
            RaySegment(tangential_vector=TangentialVector(point=s, vector=v))
            for s in ss2
        ]
        self.non_intersecting_ray_segments += [
            RaySegment(tangential_vector=TangentialVector(point=s, vector=-v))
            for s in ss2
        ]

        # convert to proper lists
        self.intersecting_rays = list(self.intersecting_rays)
        self.intersecting_ray_segments = list(self.intersecting_ray_segments)
        self.ray_depths = list(self.ray_depths)
        self.ray_segment_depths = list(self.ray_segment_depths)
        self.non_intersecting_rays = list(self.non_intersecting_rays)
        self.non_intersecting_ray_segments = list(
            self.non_intersecting_ray_segments
        )

    def test_intersetcs_ray_hits(self) -> None:
        """
        Tests if rays intersect as expected.
        """
        for ray, ray_depth in zip(self.intersecting_rays, self.ray_depths):
            for face in self.faces:
                rd = intersection_ray_depth(ray=ray, face=face)
                self.assertTrue(0 <= rd < math.inf)
                self.assertAlmostEqual(rd, ray_depth)

    def test_intersetcs_ray_segment_hits(self) -> None:
        """
        Tests if ray segments intersect as expected.
        """
        for ray, ray_depth in zip(
            self.intersecting_ray_segments, self.ray_segment_depths
        ):
            for face in self.faces:
                rd = intersection_ray_depth(ray=ray, face=face)
                self.assertTrue(0 <= rd < math.inf)
                self.assertAlmostEqual(rd, ray_depth)

    def test_intersetcs_ray_misses(self) -> None:
        """
        Tests if rays do not intersect as expected.
        """
        for ray in self.non_intersecting_rays:
            for face in self.faces:
                ray_depth = intersection_ray_depth(ray=ray, face=face)
                self.assertTrue(ray_depth == math.inf)

    def test_intersetcs_ray_segments_misses(self) -> None:
        """
        Tests if ray segments do not intersect as expected.
        """
        for ray in self.non_intersecting_ray_segments:
            for face in self.faces:
                ray_depth = intersection_ray_depth(ray=ray, face=face)
                self.assertTrue(ray_depth == math.inf)


def standard_ray_equiv(
    x: StandardGeometry.Ray, y: StandardGeometry.Ray
) -> bool:
    """Returns true iff both cartesian rays are considered equivalent."""
    return ray_segment_equiv(x.as_segment(), y.as_segment())


class StandardGeometryConstructorTest(BaseTestCase):
    def test_constructor(self) -> None:
        # pylint: disable=R0201
        """Test the constructor."""
        StandardGeometry()


class StandardGeometryRayFromTest(BaseTestCase):
    def setUp(self) -> None:
        self.geo = StandardGeometry()
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
        self.assertPredicate2(ray_segment_equiv, init_seg, self.init_seg)

    def test_ray_from_tangent(self) -> None:
        """Tests ray from tangent."""
        ray = self.geo.ray_from_tangent(self.tangent)
        init_seg = ray.as_segment()
        self.assertPredicate2(ray_segment_equiv, init_seg, self.init_seg)


class StandardGeometryIntersectsTest1(BaseTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry
        geo = StandardGeometry()
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


class StandardGeometryIntersectsTest2(BaseTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry
        geo = StandardGeometry()
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


class StandardGeometryIntersectsTest3(BaseTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry
        geo = StandardGeometry()
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


class StandardGeometryIntersectsTest4(BaseTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry
        geo = StandardGeometry()
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
