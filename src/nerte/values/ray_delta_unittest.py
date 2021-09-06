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
from nerte.values.ray import Ray
from nerte.values.ray_delta import RayDelta, ray_as_delta, add_ray_delta

# True, iff two floats are equivalent
def _equiv(x: float, y: float) -> bool:
    return math.isclose(x, y)


# True, iff two vector-like objects are equivalent
def _triple_equiv(
    x: Union[AbstractVector, Coordinates3D],
    y: Union[AbstractVector, Coordinates3D],
) -> bool:
    return _equiv(x[0], y[0]) and _equiv(x[1], y[1]) and _equiv(x[2], y[2])


# True, iff two ray-like objects are equivalent
def _ray_equiv(x: Union[Ray, RayDelta], y: Union[Ray, RayDelta]) -> bool:
    return _triple_equiv(x.start, x.start) and _triple_equiv(
        y.direction, y.direction
    )


class RayTestCase(unittest.TestCase):
    def assertEquivRay(
        self, x: Union[Ray, RayDelta], y: Union[Ray, RayDelta]
    ) -> None:
        """
        Asserts the equivalence of two ray deltas.
        Note: This replaces assertTrue(x == y) for RayDelta.
        """
        try:
            self.assertTrue(_ray_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                f"Rays or ray delta {x} is not equivalent to ray or ray delta {y}."
            ) from ae


class RayDeltaTest(RayTestCase):
    def setUp(self) -> None:
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.start = AbstractVector((0.0, 0.0, 0.0))
        self.direction = AbstractVector((1.0, 0.0, 0.0))

    def test_ray_delta(self) -> None:
        """Tests ray delta constructor."""
        ray = RayDelta(start=self.start, direction=self.direction)
        self.assertTrue(ray.start == self.start)
        self.assertTrue(ray.direction == self.direction)
        RayDelta(start=self.start, direction=self.v0)  # allowed!


class RayDeltaMathTest(RayTestCase):
    def setUp(self) -> None:
        v0 = AbstractVector((0.0, 0.0, 0.0))
        self.ray_delta0 = RayDelta(start=v0, direction=v0)
        self.ray_delta1 = RayDelta(
            start=AbstractVector((1.1, 2.2, 3.3)),
            direction=AbstractVector((5.5, 7.7, 1.1)),
        )
        self.ray_delta2 = RayDelta(
            start=AbstractVector((3.3, 6.6, 9.9)),
            direction=AbstractVector((16.5, 23.1, 3.3)),
        )
        self.ray_delta3 = RayDelta(
            start=AbstractVector((4.4, 8.8, 13.2)),
            direction=AbstractVector((22.0, 30.8, 4.4)),
        )

    def test_ray_delta_math(self) -> None:
        """Tests ray delta linear operations."""
        self.assertEquivRay(self.ray_delta1 + self.ray_delta0, self.ray_delta1)
        self.assertEquivRay(self.ray_delta1 + self.ray_delta2, self.ray_delta3)
        self.assertEquivRay(self.ray_delta3 - self.ray_delta2, self.ray_delta1)
        self.assertEquivRay(self.ray_delta1 * 3.0, self.ray_delta2)
        self.assertEquivRay(self.ray_delta2 / 3.0, self.ray_delta1)


class RayToRayDeltaConversionTest(RayTestCase):
    def setUp(self) -> None:
        v = AbstractVector((5.5, 7.7, 1.1))
        self.ray = Ray(
            start=Coordinates3D((1.1, 2.2, 3.3)),
            direction=v,
        )
        self.ray_delta = RayDelta(
            start=AbstractVector((1.1, 2.2, 3.3)),
            direction=v,
        )

    def test_ray_as_ray_delta(self) -> None:
        """Tests ray to ray delta conversion."""
        self.assertEquivRay(ray_as_delta(self.ray), self.ray_delta)


class AddRayDeltaTest(RayTestCase):
    def setUp(self) -> None:
        self.ray1 = Ray(
            start=Coordinates3D((1.0, 2.0, 3.0)),
            direction=AbstractVector((4.0, 5.0, 6.0)),
        )
        self.ray_delta = RayDelta(
            start=AbstractVector((7.0, 8.0, 9.0)),
            direction=AbstractVector((10.0, 11.0, 12.0)),
        )
        self.ray2 = Ray(
            start=Coordinates3D((8.0, 10.0, 12.0)),
            direction=AbstractVector((14.0, 16.0, 18.0)),
        )

    def test_add_ray_delta(self) -> None:
        """Tests additon of ray delta."""
        self.assertEquivRay(add_ray_delta(self.ray1, self.ray_delta), self.ray2)


if __name__ == "__main__":
    unittest.main()
