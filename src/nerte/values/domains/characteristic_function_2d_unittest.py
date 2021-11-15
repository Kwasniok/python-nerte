# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates2D
from nerte.values.domains.characteristic_function_2d import (
    CharacteristicFunction2D,
)


class ConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        def r2_func(coords: Coordinates2D) -> bool:
            return math.isfinite(coords[0]) and math.isfinite(coords[1])

        self.r2_func = r2_func

    def test_constructor(self) -> None:
        """Tests the constructor."""
        CharacteristicFunction2D(self.r2_func)


class AreInsideTest(BaseTestCase):
    def setUp(self) -> None:
        def r2_func(coords: Coordinates2D) -> bool:
            return math.isfinite(coords[0]) and math.isfinite(coords[1])

        self.coords_inside = (Coordinates2D((0.0, 0.0)),)
        self.coords_outside = (
            Coordinates2D((math.nan, 0.0)),
            Coordinates2D((0.0, math.inf)),
        )
        self.domain = CharacteristicFunction2D(r2_func)

    def test_are_inside(self) -> None:
        """Tests coordinate membership."""
        for coords in self.coords_inside:
            self.assertTrue(self.domain.are_inside(coords))
        for coords in self.coords_outside:
            self.assertFalse(self.domain.are_inside(coords))


if __name__ == "__main__":
    unittest.main()
