# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import tan_vec_equiv
from nerte.values.ray_segment import RaySegment


def ray_segment_equiv(x: RaySegment, y: RaySegment) -> bool:
    """
    Returns true iff both ray segments are considered equivalent.
    """
    return tan_vec_equiv(x.tangential_vector, x.tangential_vector) and (
        x.is_finite == y.is_finite
    )


class RaySegmentConstructorTest(BaseTestCase):
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


class RaySegmentPropertiesTest(BaseTestCase):
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
            self.assertPredicate2(
                tan_vec_equiv,
                ray.tangential_vector,
                self.tangential_vector,
            )
            self.assertTrue(ray.is_finite)
            self.assertFalse(ray.is_infinite)

        for ray in self.infinite_rays:
            self.assertPredicate2(
                tan_vec_equiv,
                ray.tangential_vector,
                self.tangential_vector,
            )
            self.assertFalse(ray.is_finite)
            self.assertTrue(ray.is_infinite)


if __name__ == "__main__":
    unittest.main()
