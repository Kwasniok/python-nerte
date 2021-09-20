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
from nerte.values.tangential_vector_unittest import (
    TangentialVectorTestCaseMixin,
)

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.tangential_vector import TangentialVector
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
            cast(
                TangentialVectorTestCaseMixin, self
            ).assertTangentialVectorEquiv(
                x.tangential_vector, y.tangential_vector
            )
            test_case.assertEqual(x.is_finite, y.is_finite)
        except AssertionError as ae:
            msg_full = f"Ray segment {x} is not equivalent to {y}."
            if msg is not None:
                msg_full += f" : {msg}"
            raise test_case.failureException(msg_full) from ae


class AssertRaySegmentEquivMixinTest(
    unittest.TestCase,
    CoordinatesTestCaseMixin,
    LinAlgTestCaseMixin,
    TangentialVectorTestCaseMixin,
    RaySegmentTestCaseMixin,
):
    def setUp(self) -> None:
        coords_0 = Coordinates3D((0.0, 0.0, 0.0))
        coords_1 = Coordinates3D((1.0, 2.0, 3.0))
        vec_1 = AbstractVector((4.0, 5.0, 6.0))
        tan_vec_0 = TangentialVector(point=coords_0, vector=vec_1)
        tan_vec_1 = TangentialVector(point=coords_1, vector=vec_1)
        self.ray_segment_0 = RaySegment(tangential_vector=tan_vec_0)
        self.ray_segment_1 = RaySegment(tangential_vector=tan_vec_1)

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
        v0 = AbstractVector((0.0, 0.0, 0.0))
        v1 = AbstractVector((1.0, 0.0, 0.0))
        point = Coordinates3D((0.0, 0.0, 0.0))
        self.tangential_vector_0 = TangentialVector(point=point, vector=v0)
        self.tangential_vector_1 = TangentialVector(point=point, vector=v1)

    def test_constructor(self) -> None:
        """Tests the constructor."""
        RaySegment(tangential_vector=self.tangential_vector_1)
        RaySegment(tangential_vector=self.tangential_vector_1, is_finite=False)
        RaySegment(tangential_vector=self.tangential_vector_1, is_finite=True)
        with self.assertRaises(ValueError):
            RaySegment(tangential_vector=self.tangential_vector_0)
        with self.assertRaises(ValueError):
            RaySegment(
                tangential_vector=self.tangential_vector_0, is_finite=False
            )
        with self.assertRaises(ValueError):
            RaySegment(
                tangential_vector=self.tangential_vector_0, is_finite=True
            )


class RaySegmentPropertiesTest(
    unittest.TestCase,
    CoordinatesTestCaseMixin,
    LinAlgTestCaseMixin,
    TangentialVectorTestCaseMixin,
    RaySegmentTestCaseMixin,
):
    def setUp(self) -> None:
        point = Coordinates3D((0.0, 0.0, 0.0))
        vector = AbstractVector((1.0, 0.0, 0.0))
        self.tangential_vector = TangentialVector(point=point, vector=vector)

        self.finite_rays = (
            RaySegment(tangential_vector=self.tangential_vector),
            RaySegment(
                tangential_vector=self.tangential_vector, is_finite=True
            ),
        )
        self.infinite_rays = (
            RaySegment(
                tangential_vector=self.tangential_vector, is_finite=False
            ),
        )

    def test_properties(self) -> None:
        """Tests the properties."""

        # preconditions
        self.assertTrue(len(self.finite_rays) > 0)
        self.assertTrue(len(self.infinite_rays) > 0)

        for ray in self.finite_rays:
            self.assertTangentialVectorEquiv(
                ray.tangential_vector, self.tangential_vector
            )
            self.assertTrue(ray.is_finite)
            self.assertFalse(ray.is_infinite)

        for ray in self.infinite_rays:
            self.assertTangentialVectorEquiv(
                ray.tangential_vector, self.tangential_vector
            )
            self.assertFalse(ray.is_finite)
            self.assertTrue(ray.is_infinite)


if __name__ == "__main__":
    unittest.main()
