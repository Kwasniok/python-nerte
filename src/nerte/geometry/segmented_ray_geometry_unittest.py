# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144


import unittest

from typing import Type, Optional

import math

from nerte.geometry.geometry_unittest import GeometryTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, normalized
from nerte.values.ray_segment import RaySegment
from nerte.values.face import Face
from nerte.values.util.convert import coordinates_as_vector
from nerte.geometry.segmented_ray_geometry import SegmentedRayGeometry


def _dummy_segmented_ray_geometry_class() -> Type[SegmentedRayGeometry]:
    class DummySegmentedRayGeometry(SegmentedRayGeometry):
        """
        Represenation of an euclidean geometry with semi-finite domain.
        """

        def __init__(self, max_steps: int, max_ray_length: float):
            SegmentedRayGeometry.__init__(self, max_steps, max_ray_length)

        def is_valid_coordinate(self, coordinates: Coordinates3D) -> bool:
            x, _, _ = coordinates
            return -1 < x < 1

        def initial_ray_segment_towards(
            self, start: Coordinates3D, target: Coordinates3D
        ) -> RaySegment:
            vec_s = coordinates_as_vector(start)
            vec_t = coordinates_as_vector(target)
            return RaySegment(start=start, direction=(vec_t - vec_s))

        def next_ray_segment(self, ray: RaySegment) -> Optional[RaySegment]:
            # old segment
            s_old = ray.start
            d_old = ray.direction
            # advance starting point
            s_new = Coordinates3D(
                (s_old[0] + d_old[0], s_old[1] + d_old[1], s_old[2] + d_old[2])
            )
            d_new = d_old
            # new segment
            if self.is_valid_coordinate(s_new):
                return RaySegment(start=s_new, direction=d_new)
            return None

        def normalize_initial_ray_segment(self, ray: RaySegment) -> RaySegment:
            return RaySegment(
                start=ray.start,
                direction=normalized(ray.direction) * self.ray_segment_length(),
            )

    return DummySegmentedRayGeometry


class SegmentedRayGeometryInterfaceTest(GeometryTestCase):
    def test_interface(self) -> None:
        # pylint: disable=R0201
        """Tests interface."""
        _dummy_segmented_ray_geometry_class()


class SegmentedRayGeometryConstructorTest(GeometryTestCase):
    def setUp(self) -> None:
        self.DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()

    def test_constructor(self) -> None:
        """Tests constructor."""
        self.DummySegmentedRayGeometry(max_steps=1, max_ray_length=1.0)
        # invalid max_step
        with self.assertRaises(ValueError):
            self.DummySegmentedRayGeometry(max_steps=0, max_ray_length=1.0)
        with self.assertRaises(ValueError):
            self.DummySegmentedRayGeometry(max_steps=-1, max_ray_length=1.0)
        # invalid max_ray_length
        with self.assertRaises(ValueError):
            self.DummySegmentedRayGeometry(max_steps=1, max_ray_length=0.0)
        with self.assertRaises(ValueError):
            self.DummySegmentedRayGeometry(max_steps=1, max_ray_length=-1.0)
        with self.assertRaises(ValueError):
            self.DummySegmentedRayGeometry(max_steps=1, max_ray_length=math.inf)
        with self.assertRaises(ValueError):
            self.DummySegmentedRayGeometry(max_steps=1, max_ray_length=math.nan)


class SegmentedRayGeometryPropertiesTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_length=1.0)

    def test_properties(self) -> None:
        """Tests properties."""
        self.assertEquiv(self.geo.ray_segment_length(), 0.1)


class SegmentedRayGeometryIsValidCoordinateTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_length=1.0)
        self.valid_coords = (Coordinates3D((0.0, 0.0, 0.0)),)
        self.invalid_coords = (
            Coordinates3D((-3.0, 0.0, 0.0)),
            Coordinates3D((3.0, 0.0, 0.0)),
            Coordinates3D((-math.inf, 0.0, 0.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D((math.nan, 0.0, 0.0)),
        )

    def test_is_valid_coordinate(self) -> None:
        """Tests coordinate validity."""
        for coords in self.valid_coords:
            self.assertTrue(self.geo.is_valid_coordinate(coords))
        for coords in self.invalid_coords:
            self.assertFalse(self.geo.is_valid_coordinate(coords))


class SegmentedRayGeometryRayTowardsTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_length=1.0)
        self.coords1 = Coordinates3D((0.0, 0.0, 0.0))
        self.coords2 = Coordinates3D((1.0, 2.0, 3.0))
        self.ray = RaySegment(
            start=self.coords1, direction=AbstractVector((1.0, 2.0, 3.0))
        )

    def test_initial_ray_segment_towards(self) -> None:
        """Tests ray towards coordinate."""
        ray = self.geo.initial_ray_segment_towards(self.coords1, self.coords2)
        self.assertCoordinates3DEquiv(ray.start, self.ray.start)
        self.assertVectorEquiv(ray.direction, self.ray.direction)


class SegmentedRayGeometryNextRaySegmentTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_length=1.0)
        direction = AbstractVector((0.75, 2.0, 3.0))
        self.ray1 = RaySegment(
            start=Coordinates3D((0.0, 0.0, 0.0)),
            direction=direction,
        )
        self.ray2 = RaySegment(
            start=Coordinates3D((0.75, 2.0, 3.0)),
            direction=direction,
        )

    def test_next_ray_segment(self) -> None:
        """Tests next ray segment."""

        ray2 = self.geo.next_ray_segment(self.ray1)
        self.assertTrue(ray2 is not None)
        if ray2 is not None:
            self.assertCoordinates3DEquiv(ray2.start, self.ray2.start)
            self.assertVectorEquiv(ray2.direction, self.ray2.direction)

        ray3 = self.geo.next_ray_segment(self.ray2)
        self.assertTrue(ray3 is None)


class SegmentedRayGeometryNormalizedInitialRayTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_length=1.0)
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


class SegmentedRayGeometryIntersectsTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_length=1.0)
        self.ray = RaySegment(
            start=Coordinates3D((0.0, 0.0, 0.0)),
            direction=AbstractVector((1.0, 1.0, 1.0)),
        )
        p0 = Coordinates3D((1.0, 0.0, 0.0))
        p1 = Coordinates3D((0.0, 1.0, 0.0))
        p2 = Coordinates3D((0.0, 0.0, 1.0))
        self.face1 = Face(p0, p1, p2)
        p0 = Coordinates3D((10.0, 0.0, 0.0))
        p1 = Coordinates3D((0.0, 10.0, 0.0))
        p2 = Coordinates3D((0.0, 0.0, 10.0))
        self.face2 = Face(p0, p1, p2)
        self.invalid_ray = RaySegment(
            start=Coordinates3D((-7.0, -7.0, -7.0)),
            direction=AbstractVector((100.0, 100.0, 100.0)),
        )

    def test_intersects(self) -> None:
        """Tests ray and face intersection."""
        info = self.geo.intersection_info(self.ray, self.face1)
        self.assertTrue(info.hits())
        info = self.geo.intersection_info(self.ray, self.face2)
        self.assertTrue(info.misses())
        with self.assertRaises(ValueError):
            self.geo.intersection_info(self.invalid_ray, self.face1)


if __name__ == "__main__":
    unittest.main()
