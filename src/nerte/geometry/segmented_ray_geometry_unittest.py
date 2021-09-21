# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import Type, Optional, cast

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
            tangent = TangentialVector(point=start, vector=(vec_t - vec_s))
            return SegmentedRayGeometry.Ray(
                geometry=self,
                initial_segment=RaySegment(tangential_vector=tangent),
            )

        def next_ray_segment(self, segment: RaySegment) -> Optional[RaySegment]:
            # old segment
            point_old = segment.tangential_vector.point
            vec_old = segment.tangential_vector.vector
            # advance pointing point
            point_new = Coordinates3D(
                (
                    point_old[0] + vec_old[0],
                    point_old[1] + vec_old[1],
                    point_old[2] + vec_old[2],
                )
            )
            vector_new = vec_old
            # new segment
            if self.is_valid_coordinate(point_new):
                return RaySegment(
                    tangential_vector=TangentialVector(
                        point=point_new, vector=vector_new
                    )
                )
            return None

        def normalize_initial_ray_segment(
            self, segment: RaySegment
        ) -> RaySegment:
            tangent = TangentialVector(
                point=segment.tangential_vector.point,
                vector=normalized(segment.tangential_vector.vector)
                * self.ray_segment_length(),
            )
            return RaySegment(tangential_vector=tangent)

    return DummySegmentedRayGeometry


class SegmentedRayGeometryImplemetationTest(BaseTestCase):
    def test_implementation(self) -> None:
        # pylint: disable=R0201
        """Tests implementation."""
        _dummy_segmented_ray_geometry_class()


class SegmentedRayGeometryConstructorTest(BaseTestCase):
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


class SegmentedRayGeometryPropertiesTest(BaseTestCase):
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
        self.assertAlmostEqual(self.geo.max_steps(), self.max_steps)
        self.assertAlmostEqual(self.geo.max_ray_depth(), self.max_ray_depth)
        self.assertAlmostEqual(
            self.geo.ray_segment_length(), self.ray_segment_length
        )


class SegmentedRayGeometryIsValidCoordinateTest(BaseTestCase):
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


class SegmentedRayGeometryRayConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_depth=1.0)
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


class SegmentedRayGeometryRayPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_depth=1.0)
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


class SegmentedRayGeometryRayIntersectsTest(BaseTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        geo = DummySegmentedRayGeometry(max_steps=10, max_ray_depth=1.0)
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


class SegmentedRayGeometryRayIntersectsRayEventuallyLeftManifoldTest(
    BaseTestCase
):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        geo = DummySegmentedRayGeometry(max_steps=10, max_ray_depth=10.0)
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


class SegmentedRayGeometryRayIntersectsRayImmediatelyLeftManifoldTest(
    BaseTestCase
):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        geo = DummySegmentedRayGeometry(max_steps=10, max_ray_depth=10.0)
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


class SegmentedRayGeometryRayIntersectsMetaDataTest(BaseTestCase):
    def setUp(self) -> None:
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.face = Face(p1, p2, p3)
        # geometry (carthesian & euclidean)
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        geos = (
            DummySegmentedRayGeometry(
                max_ray_depth=10.0,
                max_steps=10,
                # step size = 1 -> direct hit
            ),
            DummySegmentedRayGeometry(
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

    def test_runge_kutta_geometry_intersects_meta_data(self) -> None:
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


class SegmentedRayGeometryRayFromTest(BaseTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_depth=1.0)
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


class SegmentedRayGeometryNextRaySegmentTest(BaseTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_depth=1.0)
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


class SegmentedRayGeometryNormalizedInitialRayTest(BaseTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_depth=1.0)
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
