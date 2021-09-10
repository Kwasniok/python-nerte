# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest
from itertools import permutations
import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.ray import Ray
from nerte.values.face import Face
from nerte.geometry.swirl_geometry import SwirlGeometry


class SwirlGeometryTest(unittest.TestCase):
    def test_constructor(self) -> None:
        """Tests constructor."""
        SwirlGeometry(max_steps=1, max_ray_length=1.0, bend_factor=0.0)
        # invalid max_steps
        with self.assertRaises(ValueError):
            SwirlGeometry(max_steps=0, max_ray_length=1.0, bend_factor=0.0)
        # invalid ray_segment_length
        with self.assertRaises(ValueError):
            SwirlGeometry(max_steps=1, max_ray_length=0.0, bend_factor=0.0)
        with self.assertRaises(ValueError):
            SwirlGeometry(max_steps=1, max_ray_length=math.inf, bend_factor=0.0)
        with self.assertRaises(ValueError):
            SwirlGeometry(max_steps=1, max_ray_length=math.nan, bend_factor=0.0)
        # invalid bend_factor
        with self.assertRaises(ValueError):
            SwirlGeometry(max_steps=1, max_ray_length=1.0, bend_factor=math.inf)
        with self.assertRaises(ValueError):
            SwirlGeometry(max_steps=1, max_ray_length=1.0, bend_factor=math.nan)


class SwirlGeometryEuclideanEdgeCaseIntersectionTest(unittest.TestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = tuple(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry (no bend == euclidean)
        self.geo = SwirlGeometry(
            max_steps=10, max_ray_length=10.0, bend_factor=0.0
        )
        v = AbstractVector((1.0, 1.0, 1.0))
        # rays pointing 'forwards' towards faces and parallel to
        # the face's normal
        ss_hit = (
            Coordinates3D((0.0, 0.0, 0.0)),
            Coordinates3D((0.6, 0.0, 0.0)),  # one third of p1
            Coordinates3D((0.0, 0.6, 0.0)),  # one third of p2
            Coordinates3D((0.0, 0.0, 0.6)),  # one third of p3
        )
        self.intersecting_rays = tuple(
            Ray(start=s, direction=v) for s in ss_hit
        )
        # rays pointing 'forwards' towards faces and parallel to
        # the face's normal
        ss_miss = (
            Coordinates3D((-0.3, 0.3, 0.3)),
            Coordinates3D((0.3, -0.3, 0.3)),
            Coordinates3D((0.3, 0.3, -0.3)),
        )
        self.non_intersecting_rays = tuple(
            Ray(start=s, direction=v) for s in ss_miss
        )

    def test_swirl_geometry_euclidean_edge_case_intersects(self) -> None:
        """
        Tests if rays intersect as expected.
        Each ray points 'forwards' towards the face and is parallel to face's
        normal.
        """
        for r in self.intersecting_rays:
            for f in self.faces:
                info = self.geo.intersection_info(r, f)
                self.assertTrue(info.hits())
        for r in self.non_intersecting_rays:
            for f in self.faces:
                info = self.geo.intersection_info(r, f)
                self.assertTrue(info.misses())


class SwirlGeometryNonEuclideanIntersectionTest(unittest.TestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = tuple(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry (some bend == non-euclidean)
        self.geo = SwirlGeometry(
            max_steps=30, max_ray_length=10.0, bend_factor=0.5
        )
        v = AbstractVector((1.0, 1.0, 1.0))
        # NOTE: Some of the hitting and missing rays are swapped with respect to
        #       the euclidean case, because the light rays are bend.
        # rays pointing 'forwards' towards faces and initially parallel to
        # the face's normal
        ss_hit = (
            Coordinates3D((0.0, 0.0, 0.0)),
            Coordinates3D((0.6, 0.0, 0.0)),  # one third of p1
            Coordinates3D((0.0, 0.6, 0.0)),  # one third of p2
            Coordinates3D((0.0, 0.0, 0.6)),  # one third of p3
        )
        self.intersecting_rays = tuple(
            Ray(start=s, direction=v) for s in ss_hit
        )
        # rays pointing 'forwards' towards faces and initially parallel to
        # the face's normal
        ss_miss = (
            Coordinates3D((-0.3, 0.3, 0.3)),
            Coordinates3D((0.3, -0.3, 0.3)),
            Coordinates3D((0.3, 0.3, -0.3)),
        )
        self.non_intersecting_rays = tuple(
            Ray(start=s, direction=v) for s in ss_miss
        )

    def test_swirl_geometry_euclidean_edge_case_intersects(self) -> None:
        """
        Tests if rays intersect as expected.
        Each ray points 'forwards' towards the face and is parallel to face's
        normal.
        """
        for r in self.intersecting_rays:
            for f in self.faces:
                info = self.geo.intersection_info(r, f)
                self.assertTrue(info.hits())
        for r in self.non_intersecting_rays:
            for f in self.faces:
                info = self.geo.intersection_info(r, f)
                self.assertTrue(info.misses())


if __name__ == "__main__":
    unittest.main()
