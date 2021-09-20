# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import Optional, cast
from abc import ABC

from nerte.values.coordinates_unittest import CoordinatesTestCaseMixin
from nerte.values.linalg_unittest import LinAlgTestCaseMixin

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.ray_segment import RaySegment


class RaySegmentTestCaseMixin(ABC):
    # pylint: disable=R0903
    def assertRaySegmentEquiv(
        self,
        x: RaySegment,
        y: RaySegment,
        msg: Optional[str] = None,
    ) -> None:
        """
        Asserts the equivalence of two ray segments.
        """

        test_case = cast(unittest.TestCase, self)
        try:
            cast(CoordinatesTestCaseMixin, self).assertCoordinates3DEquiv(
                x.start, y.start
            )
            cast(LinAlgTestCaseMixin, self).assertVectorEquiv(
                x.direction, y.direction
            )
        except AssertionError as ae:
            msg_full = f"Ray segment {x} is not equivalent to {y}."
            if msg is not None:
                msg_full += f" : {msg}"
            raise test_case.failureException(msg_full) from ae


class AssertRaySegmentEquivMixinTest(
    unittest.TestCase,
    CoordinatesTestCaseMixin,
    LinAlgTestCaseMixin,
    RaySegmentTestCaseMixin,
):
    def setUp(self) -> None:
        coords_0 = Coordinates3D((0.0, 0.0, 0.0))
        coords_1 = Coordinates3D((1.0, 2.0, 3.0))
        vec_1 = AbstractVector((4.0, 5.0, 6.0))
        self.ray_segment_0 = RaySegment(start=coords_0, direction=vec_1)
        self.ray_segment_1 = RaySegment(start=coords_1, direction=vec_1)

    def test_ray_segment_equiv(self) -> None:
        """Tests the ray segment test case mixin."""
        self.assertRaySegmentEquiv(self.ray_segment_0, self.ray_segment_0)
        self.assertRaySegmentEquiv(self.ray_segment_1, self.ray_segment_1)

    def test_ray_segment_equiv_raise(self) -> None:
        """Tests the ray segment test case mixin raise."""
        with self.assertRaises(AssertionError):
            self.assertRaySegmentEquiv(self.ray_segment_0, self.ray_segment_1)


class RaySegmentConstructorTest(
    unittest.TestCase,
    CoordinatesTestCaseMixin,
    LinAlgTestCaseMixin,
    RaySegmentTestCaseMixin,
):
    def setUp(self) -> None:
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.start = Coordinates3D((0.0, 0.0, 0.0))
        self.direction = AbstractVector((1.0, 0.0, 0.0))

    def test_constructor(self) -> None:
        """Tests the constructor."""
        RaySegment(start=self.start, direction=self.direction)
        RaySegment(start=self.start, direction=self.direction, is_finite=False)
        RaySegment(start=self.start, direction=self.direction, is_finite=True)
        with self.assertRaises(ValueError):
            RaySegment(start=self.start, direction=self.v0)
        with self.assertRaises(ValueError):
            RaySegment(start=self.start, direction=self.v0, is_finite=False)
        with self.assertRaises(ValueError):
            RaySegment(start=self.start, direction=self.v0, is_finite=True)


class RaySegmentPropertiesTest(
    unittest.TestCase,
    CoordinatesTestCaseMixin,
    LinAlgTestCaseMixin,
    RaySegmentTestCaseMixin,
):
    def setUp(self) -> None:
        self.start = Coordinates3D((0.0, 0.0, 0.0))
        self.direction = AbstractVector((1.0, 0.0, 0.0))

        self.finite_rays = (
            RaySegment(start=self.start, direction=self.direction),
            RaySegment(
                start=self.start, direction=self.direction, is_finite=True
            ),
        )
        self.infinite_rays = (
            RaySegment(
                start=self.start, direction=self.direction, is_finite=False
            ),
        )

    def test_properties(self) -> None:
        """Tests the properties."""

        # preconditions
        self.assertTrue(len(self.finite_rays) > 0)
        self.assertTrue(len(self.infinite_rays) > 0)

        for ray in self.finite_rays:
            self.assertCoordinates3DEquiv(ray.start, self.start)
            self.assertVectorEquiv(ray.direction, self.direction)
            self.assertTrue(ray.is_finite)
            self.assertFalse(ray.is_infinite)

        for ray in self.infinite_rays:
            self.assertCoordinates3DEquiv(ray.start, self.start)
            self.assertVectorEquiv(ray.direction, self.direction)
            self.assertFalse(ray.is_finite)
            self.assertTrue(ray.is_infinite)


if __name__ == "__main__":
    unittest.main()
