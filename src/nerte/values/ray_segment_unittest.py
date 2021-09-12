# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import Union

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.ray_segment import RaySegment


# True, iff two floats are equivalent
def _equiv(x: float, y: float) -> bool:
    return math.isclose(x, y)


# True, iff two vector-like objects are equivalent
def _triple_equiv(
    x: Union[AbstractVector, Coordinates3D],
    y: Union[AbstractVector, Coordinates3D],
) -> bool:
    return _equiv(x[0], y[0]) and _equiv(x[1], y[1]) and _equiv(x[2], y[2])


class RaySegmentTestCase(unittest.TestCase):
    def assertEquivVec(self, x: AbstractVector, y: AbstractVector) -> None:
        """
        Asserts the equivalence of two vectors.
        Note: This replaces assertTrue(x == y) for vectors.
        """
        try:
            self.assertTrue(_triple_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                f"Vector {x} is not equivalent to  {y}."
            ) from ae

    def assertEquivCoords(self, x: Coordinates3D, y: Coordinates3D) -> None:
        """
        Asserts the equivalence of two coordinates.
        Note: This replaces assertTrue(x == y) for coordinates.
        """
        try:
            self.assertTrue(_triple_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                f"Coordinates {x} are not equivalent to  {y}."
            ) from ae


class RaySegmentConstructorTest(RaySegmentTestCase):
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


class RaySegmentPropertiesTest(RaySegmentTestCase):
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
            self.assertEquivCoords(ray.start, self.start)
            self.assertEquivVec(ray.direction, self.direction)
            self.assertTrue(ray.is_finite)
            self.assertFalse(ray.is_infinite)

        for ray in self.infinite_rays:
            self.assertEquivCoords(ray.start, self.start)
            self.assertEquivVec(ray.direction, self.direction)
            self.assertFalse(ray.is_finite)
            self.assertTrue(ray.is_infinite)


if __name__ == "__main__":
    unittest.main()
