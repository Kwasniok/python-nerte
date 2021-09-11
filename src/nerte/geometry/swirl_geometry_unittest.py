# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest
from itertools import permutations
import math

from nerte.geometry.geometry_unittest import GeometryTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, normalized
from nerte.values.ray_segment import RaySegment
from nerte.values.face import Face
from nerte.values.util.convert import vector_as_coordinates
from nerte.geometry.swirl_geometry import SwirlGeometry


class SwirlGeometryConstructorTest(GeometryTestCase):
    def test_constructor(self) -> None:
        """Tests constructor."""
        SwirlGeometry(max_steps=1, max_ray_depth=1.0, bend_factor=0.0)
        # invalid max_steps
        with self.assertRaises(ValueError):
            SwirlGeometry(max_steps=0, max_ray_depth=1.0, bend_factor=0.0)
        # invalid ray_segment_length
        with self.assertRaises(ValueError):
            SwirlGeometry(max_steps=1, max_ray_depth=0.0, bend_factor=0.0)
        with self.assertRaises(ValueError):
            SwirlGeometry(max_steps=1, max_ray_depth=math.inf, bend_factor=0.0)
        with self.assertRaises(ValueError):
            SwirlGeometry(max_steps=1, max_ray_depth=math.nan, bend_factor=0.0)
        # invalid bend_factor
        with self.assertRaises(ValueError):
            SwirlGeometry(max_steps=1, max_ray_depth=1.0, bend_factor=math.inf)
        with self.assertRaises(ValueError):
            SwirlGeometry(max_steps=1, max_ray_depth=1.0, bend_factor=math.nan)


class SwirlGeometryPropertiesTest(GeometryTestCase):
    def setUp(self) -> None:
        self.max_steps = 10
        self.max_ray_depth = 1.0
        self.ray_segment_length = self.max_ray_depth / self.max_steps
        self.bend_factor = 3.3
        self.geo = SwirlGeometry(
            max_steps=self.max_steps,
            max_ray_depth=self.max_ray_depth,
            bend_factor=self.bend_factor,
        )

    def test_properties(self) -> None:
        """Tests properties."""
        self.assertEquiv(self.geo.max_steps(), self.max_steps)
        self.assertEquiv(self.geo.max_ray_depth(), self.max_ray_depth)
        self.assertEquiv(self.geo.ray_segment_length(), self.ray_segment_length)
        self.assertEquiv(self.geo.bend_factor, self.bend_factor)


class SwirlGeometryIsValidCoordinateTest(GeometryTestCase):
    def setUp(self) -> None:
        self.geo = SwirlGeometry(
            max_steps=10, max_ray_depth=1.0, bend_factor=0.0
        )
        self.valid_coords = (Coordinates3D((0.0, 0.0, 0.0)),)

    def test_is_valid_coordinate(self) -> None:
        """Tests coordinate validity."""
        for coords in self.valid_coords:
            self.assertTrue(self.geo.is_valid_coordinate(coords))


class SwirlGeometryRayFromTest(GeometryTestCase):
    def setUp(self) -> None:
        self.geo = SwirlGeometry(
            max_steps=10, max_ray_depth=1.0, bend_factor=0.0
        )
        self.coords1 = Coordinates3D((0.0, 0.0, 0.0))
        self.coords2 = Coordinates3D((0.0, 1.0, 2.0))
        self.direction = AbstractVector((0.0, 1.0, 2.0))  # equiv to cords2
        self.init_seg = self.geo.normalize_initial_ray_segment(
            RaySegment(start=self.coords1, direction=self.direction)
        )

    def test_ray_from_coords(self) -> None:
        """Tests ray from coordinates."""
        ray = self.geo.ray_from_coords(self.coords1, self.coords2)
        init_seg = ray.initial_segment()
        self.assertCoordinates3DEquiv(init_seg.start, self.init_seg.start)
        self.assertVectorEquiv(init_seg.direction, self.init_seg.direction)

    def test_ray_from_tangent(self) -> None:
        """Tests ray from tangent."""
        ray = self.geo.ray_from_tangent(self.coords1, self.direction)
        init_seg = ray.initial_segment()
        self.assertCoordinates3DEquiv(init_seg.start, self.init_seg.start)
        self.assertVectorEquiv(init_seg.direction, self.init_seg.direction)


class SwirlGeometryNextRaySegmentTest(GeometryTestCase):
    def setUp(self) -> None:
        self.geo = SwirlGeometry(
            max_steps=1, max_ray_depth=1.0, bend_factor=0.0
        )
        direction = AbstractVector((0.75, 2.0, 3.0))
        self.ray1 = RaySegment(
            start=Coordinates3D((0.0, 0.0, 0.0)),
            direction=direction,
        )
        self.ray2 = RaySegment(
            start=vector_as_coordinates(direction),
            direction=normalized(direction) * self.geo.ray_segment_length(),
        )

    def test_next_ray_segment(self) -> None:
        """Tests next ray segment."""

        ray2 = self.geo.next_ray_segment(self.ray1)
        self.assertCoordinates3DEquiv(ray2.start, self.ray2.start)
        self.assertVectorEquiv(ray2.direction, self.ray2.direction)


class SwirlGeometryNormalizedInitialRayTest(GeometryTestCase):
    def setUp(self) -> None:
        self.geo = SwirlGeometry(
            max_steps=10, max_ray_depth=1.0, bend_factor=0.0
        )
        corrds0 = Coordinates3D((0.0, 0.0, 0.0))
        self.ray = RaySegment(
            start=corrds0,
            direction=AbstractVector((1.0, 2.0, 3.0)),
        )
        self.ray_normalized = RaySegment(
            start=corrds0,
            direction=normalized(AbstractVector((1.0, 2.0, 3.0)))
            * self.geo.ray_segment_length(),
        )

    def test_normalize_initial_ray_segment(self) -> None:
        """Tests normalized initial ray segment."""
        ray = self.geo.normalize_initial_ray_segment(self.ray)
        self.assertTrue(ray is not None)
        if ray is not None:
            self.assertCoordinates3DEquiv(ray.start, self.ray_normalized.start)
            self.assertVectorEquiv(ray.direction, self.ray_normalized.direction)


class SwirlGeometryEuclideanEdgeCaseIntersectionTest(GeometryTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = tuple(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry (no bend == euclidean)
        geo = SwirlGeometry(max_steps=10, max_ray_depth=10.0, bend_factor=0.0)
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
            geo.ray_from_tangent(start=s, direction=v) for s in ss_hit
        )
        # rays pointing 'forwards' towards faces and parallel to
        # the face's normal
        ss_miss = (
            Coordinates3D((-0.3, 0.3, 0.3)),
            Coordinates3D((0.3, -0.3, 0.3)),
            Coordinates3D((0.3, 0.3, -0.3)),
        )
        self.non_intersecting_rays = tuple(
            geo.ray_from_tangent(start=s, direction=v) for s in ss_miss
        )

    def test_swirl_geometry_euclidean_edge_case_intersects(self) -> None:
        """
        Tests if rays intersect as expected.
        Each ray points 'forwards' towards the face and is parallel to face's
        normal.
        """
        for r in self.intersecting_rays:
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.hits())
        for r in self.non_intersecting_rays:
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.misses())


class SwirlGeometryNonEuclideanIntersectionTest(GeometryTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = tuple(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry (some bend == non-euclidean)
        geo = SwirlGeometry(max_steps=30, max_ray_depth=10.0, bend_factor=0.5)
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
            geo.ray_from_tangent(start=s, direction=v) for s in ss_hit
        )
        # rays pointing 'forwards' towards faces and initially parallel to
        # the face's normal
        ss_miss = (
            Coordinates3D((-0.3, 0.3, 0.3)),
            Coordinates3D((0.3, -0.3, 0.3)),
            Coordinates3D((0.3, 0.3, -0.3)),
        )
        self.non_intersecting_rays = tuple(
            geo.ray_from_tangent(start=s, direction=v) for s in ss_miss
        )

    def test_swirl_geometry_euclidean_edge_case_intersects(self) -> None:
        """
        Tests if rays intersect as expected.
        Each ray points 'forwards' towards the face and is parallel to face's
        normal.
        """
        for r in self.intersecting_rays:
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.hits())
        for r in self.non_intersecting_rays:
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.misses())


if __name__ == "__main__":
    unittest.main()
