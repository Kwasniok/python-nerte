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
from nerte.values.intersection_info import IntersectionInfo
from nerte.values.util.convert import coordinates_as_vector
from nerte.geometry.segmented_ray_geometry import SegmentedRayGeometry


def _dummy_segmented_ray_geometry_class() -> Type[SegmentedRayGeometry]:
    class DummySegmentedRayGeometry(SegmentedRayGeometry):
        """
        Represenation of an euclidean geometry with semi-finite domain.
        """

        def __init__(self, max_steps: int, max_ray_depth: float):
            SegmentedRayGeometry.__init__(self, max_steps, max_ray_depth)

        def is_valid_coordinate(self, coordinates: Coordinates3D) -> bool:
            x, _, _ = coordinates
            return -1 < x < 1

        def ray_from_coords(
            self, start: Coordinates3D, target: Coordinates3D
        ) -> SegmentedRayGeometry.Ray:
            if not self.is_valid_coordinate(start):
                raise ValueError(
                    f"Cannot create ray from coordinates."
                    f" Start coordinates {start} are invalid."
                )
            if not self.is_valid_coordinate(target):
                raise ValueError(
                    f"Cannot create ray from coordinates."
                    f" Target coordinates {target} are invalid."
                )
            vec_s = coordinates_as_vector(start)
            vec_t = coordinates_as_vector(target)
            return SegmentedRayGeometry.Ray(
                geometry=self,
                initial_segment=RaySegment(
                    start=start, direction=(vec_t - vec_s)
                ),
            )

        def next_ray_segment(self, segment: RaySegment) -> Optional[RaySegment]:
            # old segment
            s_old = segment.start
            d_old = segment.direction
            # advance starting point
            s_new = Coordinates3D(
                (s_old[0] + d_old[0], s_old[1] + d_old[1], s_old[2] + d_old[2])
            )
            d_new = d_old
            # new segment
            if self.is_valid_coordinate(s_new):
                return RaySegment(start=s_new, direction=d_new)
            return None

        def normalize_initial_ray_segment(
            self, segment: RaySegment
        ) -> RaySegment:
            return RaySegment(
                start=segment.start,
                direction=normalized(segment.direction)
                * self.ray_segment_length(),
            )

    return DummySegmentedRayGeometry


class SegmentedRayGeometryImplemetationTest(GeometryTestCase):
    def test_implementation(self) -> None:
        # pylint: disable=R0201
        """Tests implementation."""
        _dummy_segmented_ray_geometry_class()


class SegmentedRayGeometryConstructorTest(GeometryTestCase):
    def setUp(self) -> None:
        self.DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()

    def test_constructor(self) -> None:
        """Tests constructor."""
        self.DummySegmentedRayGeometry(max_steps=1, max_ray_depth=1.0)
        # invalid max_step
        with self.assertRaises(ValueError):
            self.DummySegmentedRayGeometry(max_steps=0, max_ray_depth=1.0)
        with self.assertRaises(ValueError):
            self.DummySegmentedRayGeometry(max_steps=-1, max_ray_depth=1.0)
        # invalid max_ray_depth
        with self.assertRaises(ValueError):
            self.DummySegmentedRayGeometry(max_steps=1, max_ray_depth=0.0)
        with self.assertRaises(ValueError):
            self.DummySegmentedRayGeometry(max_steps=1, max_ray_depth=-1.0)
        with self.assertRaises(ValueError):
            self.DummySegmentedRayGeometry(max_steps=1, max_ray_depth=math.inf)
        with self.assertRaises(ValueError):
            self.DummySegmentedRayGeometry(max_steps=1, max_ray_depth=math.nan)


class SegmentedRayGeometryPropertiesTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.max_steps = 10
        self.max_ray_depth = 1.0
        self.ray_segment_length = self.max_ray_depth / self.max_steps
        self.geo = DummySegmentedRayGeometry(
            max_steps=self.max_steps, max_ray_depth=self.max_ray_depth
        )

    def test_properties(self) -> None:
        """Tests properties."""
        self.assertEquiv(self.geo.max_steps(), self.max_steps)
        self.assertEquiv(self.geo.max_ray_depth(), self.max_ray_depth)
        self.assertEquiv(self.geo.ray_segment_length(), self.ray_segment_length)


class SegmentedRayGeometryIsValidCoordinateTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_depth=1.0)
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


class SegmentedRayGeometryRayConstructorTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_depth=1.0)
        self.coords = Coordinates3D((0.0, 0.0, 0.0))
        self.direction = AbstractVector((0.0, 1.0, 2.0))
        self.initial_segment = self.geo.normalize_initial_ray_segment(
            RaySegment(start=self.coords, direction=self.direction)
        )

    def test_constructor(self) -> None:
        """Tests the constructor."""
        SegmentedRayGeometry.Ray(
            geometry=self.geo, initial_segment=self.initial_segment
        )


class SegmentedRayGeometryRayPropertiesTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_depth=1.0)
        coords = Coordinates3D((0.0, 0.0, 0.0))
        direction = AbstractVector((0.0, 1.0, 2.0))
        self.ray = SegmentedRayGeometry.Ray(
            geometry=self.geo,
            initial_segment=RaySegment(start=coords, direction=direction),
        )
        self.initial_segment = self.geo.normalize_initial_ray_segment(
            RaySegment(start=coords, direction=direction)
        )

    def test_properties(self) -> None:
        """Tests the properties."""
        self.assertEquivRaySegment(
            self.ray.initial_segment(), self.initial_segment
        )


class SegmentedRayGeometryRayIntersectsTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        geo = DummySegmentedRayGeometry(max_steps=10, max_ray_depth=1.0)
        self.ray = geo.ray_from_tangent(
            start=Coordinates3D((0.0, 0.0, 0.0)),
            direction=AbstractVector((1.0, 1.0, 1.0)),
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
        self.assertFalse(
            IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD in info.miss_reasons()
        )


class SegmentedRayGeometryRayIntersectsRayEventuallyLeftManifoldTest(
    GeometryTestCase
):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        geo = DummySegmentedRayGeometry(max_steps=10, max_ray_depth=10.0)
        self.ray = geo.ray_from_tangent(
            start=Coordinates3D((0.0, 0.0, 0.0)),
            direction=AbstractVector((1.0, 0.0, 0.0)),
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
            IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD in info.miss_reasons()
        )


class SegmentedRayGeometryRayIntersectsRayImmediatelyLeftManifoldTest(
    GeometryTestCase
):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        geo = DummySegmentedRayGeometry(max_steps=10, max_ray_depth=10.0)
        self.ray = geo.ray_from_tangent(
            start=Coordinates3D(
                (0.99, 0.0, 0.0)  # close to manifold boundaries
            ),
            direction=AbstractVector((1.0, 0.0, 0.0)),
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
            IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD in info.miss_reasons()
        )


class SegmentedRayGeometryRayFromTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_depth=1.0)
        self.coords1 = Coordinates3D((0.0, 0.0, 0.0))
        self.coords2 = Coordinates3D((0.0, 1.0, 2.0))
        self.invalid_coords = Coordinates3D((-3.0, 0.0, 0.0))
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
        with self.assertRaises(ValueError):
            self.geo.ray_from_coords(self.invalid_coords, self.coords2)
        with self.assertRaises(ValueError):
            self.geo.ray_from_coords(self.coords1, self.invalid_coords)
        with self.assertRaises(ValueError):
            self.geo.ray_from_coords(self.invalid_coords, self.invalid_coords)

    def test_ray_from_tangent(self) -> None:
        """Tests ray from tangent."""
        ray = self.geo.ray_from_tangent(self.coords1, self.direction)
        init_seg = ray.initial_segment()
        self.assertCoordinates3DEquiv(init_seg.start, self.init_seg.start)
        self.assertVectorEquiv(init_seg.direction, self.init_seg.direction)
        with self.assertRaises(ValueError):
            self.geo.ray_from_tangent(self.invalid_coords, self.direction)


class SegmentedRayGeometryNextRaySegmentTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_depth=1.0)
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
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_depth=1.0)
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


if __name__ == "__main__":
    unittest.main()
