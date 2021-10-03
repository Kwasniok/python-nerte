# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import cast

import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, normalized
from nerte.values.tangential_vector import TangentialVector
from nerte.values.ray_segment import RaySegment
from nerte.values.ray_segment_unittest import ray_segment_equiv
from nerte.values.face import Face
from nerte.values.intersection_info import IntersectionInfo
from nerte.values.extended_intersection_info import ExtendedIntersectionInfo
from nerte.values.interval import Interval, REALS
from nerte.values.domains import CartesianProduct3D
from nerte.values.charts import IdentityChart3D
from nerte.geometry.segmented_ray_geometry import SegmentedRayGeometry


class ConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.chart = IdentityChart3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )

    def test_constructor(self) -> None:
        """Tests constructor."""
        SegmentedRayGeometry(self.chart, max_steps=1, max_ray_depth=1.0)
        # invalid max_step
        with self.assertRaises(ValueError):
            SegmentedRayGeometry(self.chart, max_steps=0, max_ray_depth=1.0)
        with self.assertRaises(ValueError):
            SegmentedRayGeometry(self.chart, max_steps=-1, max_ray_depth=1.0)
        # invalid max_ray_depth
        with self.assertRaises(ValueError):
            SegmentedRayGeometry(self.chart, max_steps=1, max_ray_depth=0.0)
        with self.assertRaises(ValueError):
            SegmentedRayGeometry(self.chart, max_steps=1, max_ray_depth=-1.0)
        with self.assertRaises(ValueError):
            SegmentedRayGeometry(
                self.chart, max_steps=1, max_ray_depth=math.inf
            )
        with self.assertRaises(ValueError):
            SegmentedRayGeometry(
                self.chart, max_steps=1, max_ray_depth=math.nan
            )


class PropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        chart = IdentityChart3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        self.max_steps = 10
        self.max_ray_depth = 1.0
        self.ray_segment_length = self.max_ray_depth / self.max_steps
        self.geo = SegmentedRayGeometry(
            chart, max_steps=self.max_steps, max_ray_depth=self.max_ray_depth
        )

    def test_properties(self) -> None:
        """Tests properties."""
        self.assertAlmostEqual(self.geo.max_steps(), self.max_steps)
        self.assertAlmostEqual(self.geo.max_ray_depth(), self.max_ray_depth)
        self.assertAlmostEqual(
            self.geo.ray_segment_length(), self.ray_segment_length
        )


class AreValidCoordinatesTest(BaseTestCase):
    def setUp(self) -> None:
        chart = IdentityChart3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        self.geo = SegmentedRayGeometry(chart, max_steps=10, max_ray_depth=1.0)
        self.valid_coords = (Coordinates3D((0.0, 0.0, 0.0)),)
        self.invalid_coords = (
            Coordinates3D((-3.0, 0.0, 0.0)),
            Coordinates3D((3.0, 0.0, 0.0)),
            Coordinates3D((-math.inf, 0.0, 0.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D((math.nan, 0.0, 0.0)),
        )

    def test_are_valid_coords(self) -> None:
        """Tests coordinate validity."""
        for coords in self.valid_coords:
            self.assertTrue(self.geo.are_valid_coords(coords))
        for coords in self.invalid_coords:
            self.assertFalse(self.geo.are_valid_coords(coords))


class RayConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        chart = IdentityChart3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        self.geo = SegmentedRayGeometry(chart, max_steps=10, max_ray_depth=1.0)
        self.coords = Coordinates3D((0.0, 0.0, 0.0))
        self.vector = AbstractVector((0.0, 1.0, 2.0))
        self.tangent = TangentialVector(point=self.coords, vector=self.vector)
        self.initial_segment = self.geo.normalize_initial_ray_segment(
            RaySegment(tangential_vector=self.tangent)
        )

    def test_constructor(self) -> None:
        """Tests the constructor."""
        SegmentedRayGeometry.Ray(
            geometry=self.geo, initial_segment=self.initial_segment
        )


class RayPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        chart = IdentityChart3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        self.geo = SegmentedRayGeometry(chart, max_steps=10, max_ray_depth=1.0)
        coords = Coordinates3D((0.0, 0.0, 0.0))
        vector = AbstractVector((0.0, 1.0, 2.0))
        tangent = TangentialVector(point=coords, vector=vector)
        self.ray = SegmentedRayGeometry.Ray(
            geometry=self.geo,
            initial_segment=RaySegment(tangential_vector=tangent),
        )
        self.initial_segment = self.geo.normalize_initial_ray_segment(
            RaySegment(tangential_vector=tangent),
        )

    def test_properties(self) -> None:
        """Tests the properties."""
        self.assertPredicate2(
            ray_segment_equiv, self.ray.initial_segment(), self.initial_segment
        )


class RayIntersectsTest(BaseTestCase):
    def setUp(self) -> None:
        chart = IdentityChart3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        geo = SegmentedRayGeometry(chart, max_steps=10, max_ray_depth=1.0)
        self.ray = geo.ray_from_tangent(
            TangentialVector(
                point=Coordinates3D((0.0, 0.0, 0.0)),
                vector=AbstractVector((1.0, 1.0, 1.0)),
            )
        )
        p0 = Coordinates3D((1.0, 0.0, 0.0))
        p1 = Coordinates3D((0.0, 1.0, 0.0))
        p2 = Coordinates3D((0.0, 0.0, 1.0))
        self.face_near = Face(p0, p1, p2)
        p0 = Coordinates3D((10.0, 0.0, 0.0))
        p1 = Coordinates3D((0.0, 10.0, 0.0))
        p2 = Coordinates3D((0.0, 0.0, 10.0))
        self.face_far = Face(p0, p1, p2)

    def test_hits_near_face(self) -> None:
        """Tests ray and face intersection."""
        info = self.ray.intersection_info(self.face_near)
        self.assertTrue(info.hits())

    def test_misses_far_face(self) -> None:
        """Tests ray and face intersection."""
        info = self.ray.intersection_info(self.face_far)
        self.assertTrue(info.misses())
        self.assertTrue(
            info.has_miss_reason(IntersectionInfo.MissReason.NO_INTERSECTION)
        )


class RayIntersectsRayEventuallyLeftManifoldTest(BaseTestCase):
    def setUp(self) -> None:
        chart = IdentityChart3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        geo = SegmentedRayGeometry(chart, max_steps=10, max_ray_depth=10.0)
        self.ray = geo.ray_from_tangent(
            TangentialVector(
                point=Coordinates3D((0.0, 0.0, 0.0)),
                vector=AbstractVector((1.0, 0.0, 0.0)),
            )
        )
        p0 = Coordinates3D((1.0, 0.0, 1.0))
        p1 = Coordinates3D((1.0, 1.0, 1.0))
        p2 = Coordinates3D((0.0, 1.0, 1.0))
        self.face = Face(p0, p1, p2)

    def test_ray_leaves_manifold_eventually(self) -> None:
        """
        Tests ray and face intersection where some ray segment points outside
        of the manifolds baounadires but not the first one.
        """
        info = self.ray.intersection_info(self.face)
        self.assertTrue(info.misses())
        self.assertTrue(
            info.has_miss_reason(IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD)
        )


class RayIntersectsRayImmediatelyLeftManifoldTest(BaseTestCase):
    def setUp(self) -> None:
        chart = IdentityChart3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        geo = SegmentedRayGeometry(chart, max_steps=10, max_ray_depth=10.0)
        self.ray = geo.ray_from_tangent(
            TangentialVector(
                point=Coordinates3D(
                    (0.99, 0.0, 0.0)  # close to manifold boundaries
                ),
                vector=AbstractVector((1.0, 0.0, 0.0)),
            )
        )
        p0 = Coordinates3D((1.0, 0.0, 1.0))
        p1 = Coordinates3D((1.0, 1.0, 1.0))
        p2 = Coordinates3D((0.0, 1.0, 1.0))
        self.face = Face(p0, p1, p2)

    def test_ray_leaves_manifold_immediately(self) -> None:
        """
        Tests ray and face intersection where first ray segment points outside
        of the manifolds baounadires.
        """
        info = self.ray.intersection_info(self.face)
        self.assertTrue(info.misses())
        self.assertTrue(
            info.has_miss_reason(IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD)
        )


class RayIntersectsMetaDataTest(BaseTestCase):
    def setUp(self) -> None:
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.face = Face(p1, p2, p3)
        # geometry (cartesian & euclidean)
        chart = IdentityChart3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        geos = (
            SegmentedRayGeometry(
                chart,
                max_ray_depth=10.0,
                max_steps=10,
                # step size = 1 -> direct hit
            ),
            SegmentedRayGeometry(
                chart,
                max_ray_depth=10.0,
                max_steps=100,
                # step size = 0.1 -> 6 steps until hit (1/sqrt(3) ~ 0.577...)
            ),
        )
        self.rays = tuple(
            geo.ray_from_tangent(
                TangentialVector(
                    point=Coordinates3D((0.0, 0.0, 0.0)),
                    vector=AbstractVector((1.0, 1.0, 1.0)),
                )
            )
            for geo in geos
        )
        self.steps = (1, 6)

    def test_intersects_meta_data(self) -> None:
        """
        Tests if ray's meta data.
        """
        for ray, steps in zip(self.rays, self.steps):
            info = ray.intersection_info(self.face)
            self.assertIsInstance(info, ExtendedIntersectionInfo)
            if isinstance(info, ExtendedIntersectionInfo):
                info = cast(ExtendedIntersectionInfo, info)
                meta_data = info.meta_data
                self.assertIsNotNone(meta_data)
                if meta_data is not None:
                    self.assertTrue("steps" in meta_data)
                    self.assertAlmostEqual(meta_data["steps"], steps)


class RayFromTest(BaseTestCase):
    def setUp(self) -> None:
        chart = IdentityChart3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        self.geo = SegmentedRayGeometry(chart, max_steps=10, max_ray_depth=1.0)
        self.coords1 = Coordinates3D((0.0, 0.0, 0.0))
        self.coords2 = Coordinates3D((0.0, 1.0, 2.0))
        self.invalid_coords = Coordinates3D((-3.0, 0.0, 0.0))
        vector = AbstractVector((0.0, 1.0, 2.0))  # equiv to cords2
        self.tangent = TangentialVector(point=self.coords1, vector=vector)
        self.invalid_tangent = TangentialVector(
            point=self.invalid_coords, vector=vector
        )
        self.init_seg = self.geo.normalize_initial_ray_segment(
            RaySegment(
                tangential_vector=TangentialVector(
                    point=self.coords1, vector=vector
                )
            )
        )

    def test_ray_from_coords(self) -> None:
        """Tests ray from coordinates."""
        ray = self.geo.ray_from_coords(self.coords1, self.coords2)
        init_seg = ray.initial_segment()
        self.assertPredicate2(ray_segment_equiv, init_seg, self.init_seg)
        with self.assertRaises(ValueError):
            self.geo.ray_from_coords(self.invalid_coords, self.coords2)
        with self.assertRaises(ValueError):
            self.geo.ray_from_coords(self.coords1, self.invalid_coords)
        with self.assertRaises(ValueError):
            self.geo.ray_from_coords(self.invalid_coords, self.invalid_coords)

    def test_ray_from_tangent(self) -> None:
        """Tests ray from tangent."""
        ray = self.geo.ray_from_tangent(self.tangent)
        init_seg = ray.initial_segment()
        self.assertPredicate2(ray_segment_equiv, init_seg, self.init_seg)
        with self.assertRaises(ValueError):
            self.geo.ray_from_tangent(self.invalid_tangent)


class NextRaySegmentTest(BaseTestCase):
    def setUp(self) -> None:
        chart = IdentityChart3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        self.geo = SegmentedRayGeometry(chart, max_steps=10, max_ray_depth=1.0)
        vector = AbstractVector((0.75, 2.0, 3.0))
        self.ray1 = RaySegment(
            tangential_vector=TangentialVector(
                point=Coordinates3D((0.0, 0.0, 0.0)),
                vector=vector,
            )
        )
        self.ray2 = RaySegment(
            tangential_vector=TangentialVector(
                point=Coordinates3D((0.75, 2.0, 3.0)),
                vector=vector,
            )
        )

    def test_next_ray_segment(self) -> None:
        """Tests next ray segment."""

        ray2 = self.geo.next_ray_segment(self.ray1)
        self.assertTrue(ray2 is not None)
        if ray2 is not None:
            self.assertPredicate2(ray_segment_equiv, ray2, self.ray2)

        ray3 = self.geo.next_ray_segment(self.ray2)
        self.assertTrue(ray3 is None)


class NormalizedInitialRayTest(BaseTestCase):
    def setUp(self) -> None:
        chart = IdentityChart3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        self.geo = SegmentedRayGeometry(chart, max_steps=10, max_ray_depth=1.0)
        corrds0 = Coordinates3D((0.0, 0.0, 0.0))
        self.ray = RaySegment(
            tangential_vector=TangentialVector(
                point=corrds0,
                vector=AbstractVector((1.0, 2.0, 3.0)),
            )
        )
        self.ray_normalized = RaySegment(
            tangential_vector=TangentialVector(
                point=corrds0,
                vector=normalized(AbstractVector((1.0, 2.0, 3.0)))
                * self.geo.ray_segment_length(),
            )
        )

    def test_normalize_initial_ray_segment(self) -> None:
        """Tests normalized initial ray segment."""
        ray = self.geo.normalize_initial_ray_segment(self.ray)
        self.assertTrue(ray is not None)
        if ray is not None:
            self.assertPredicate2(ray_segment_equiv, ray, self.ray_normalized)


if __name__ == "__main__":
    unittest.main()
