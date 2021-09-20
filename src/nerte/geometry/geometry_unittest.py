# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144


import unittest

from itertools import permutations
import math

from nerte.values.coordinates_unittest import CoordinatesTestCaseMixin
from nerte.values.linalg_unittest import LinAlgTestCaseMixin
from nerte.values.ray_segment_unittest import RaySegmentTestCaseMixin
from nerte.values.ray_segment_delta_unittest import RaySegmentDeltaTestCaseMixin

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.ray_segment import RaySegment
from nerte.values.face import Face
from nerte.geometry.geometry import intersection_ray_depth


# no test for abstract class/interface Geometry & Geometry.Ray

# TODO: add dedicated tests for intersection infos where ray leaves the manifold


class GeometryTestCaseMixin(
    CoordinatesTestCaseMixin,
    LinAlgTestCaseMixin,
    RaySegmentTestCaseMixin,
    RaySegmentDeltaTestCaseMixin,
):
    pass


class IntersectionRayDepthTest(unittest.TestCase, GeometryTestCaseMixin):
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
            RaySegment(start=s, direction=v * 0.1, is_finite=False) for s in ss1
        ]
        self.ray_depths = [10 / 3, 20 / 9, 20 / 9, 20 / 9]
        self.intersecting_ray_segments = [
            RaySegment(start=s, direction=v * 1.0) for s in ss1
        ]
        self.ray_segment_depths = [1 / 3, 2 / 9, 2 / 9, 2 / 9]
        self.non_intersecting_ray_segments = [
            RaySegment(start=s, direction=v * 0.1) for s in ss1
        ]
        # rays pointing 'backwards'
        # away from the face and parallel to face normal
        self.non_intersecting_rays = [
            RaySegment(start=s, direction=-v, is_finite=False) for s in ss1
        ]
        self.non_intersecting_ray_segments += [
            RaySegment(start=s, direction=-v) for s in ss1
        ]
        s21 = Coordinates3D((0.0, 0.6, 0.6))  # 'complement' of p1
        s22 = Coordinates3D((0.6, 0.0, 0.6))  # 'complement' of p2
        s23 = Coordinates3D((0.6, 0.6, 0.0))  # 'complement' of p3
        ss2 = (s21, s22, s23)
        # rays parallel to face normal but starting 'outside' the face
        self.non_intersecting_rays += [
            RaySegment(start=s, direction=v, is_finite=False) for s in ss2
        ]
        self.non_intersecting_rays += [
            RaySegment(start=s, direction=-v, is_finite=False) for s in ss2
        ]
        self.non_intersecting_ray_segments += [
            RaySegment(start=s, direction=v) for s in ss2
        ]
        self.non_intersecting_ray_segments += [
            RaySegment(start=s, direction=-v) for s in ss2
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


if __name__ == "__main__":
    unittest.main()
