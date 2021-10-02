# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates2D
from nerte.values.domains.base import OutOfDomainError
from nerte.values.domains.domain_2d import R2, Empty


class R2PropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.valid_coords = (Coordinates2D((1, 2)),)
        self.invalid_coords = (
            Coordinates2D((math.inf, 2)),
            Coordinates2D((1, -math.inf)),
            Coordinates2D((1, math.nan)),
        )
        self.r2 = R2

    def test_properties_coords_valid(self) -> None:
        """Tests domain properties when coordinates are valid."""
        for coords in self.valid_coords:
            self.assertTrue(self.r2.are_inside(coords))

    def test_properties_coords_invalid(self) -> None:
        """Tests domain properties when coordinates are valid."""
        for coords in self.invalid_coords:
            self.assertFalse(self.r2.are_inside(coords))


class EmptyPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.invalid_coords = (
            Coordinates2D((1, 2)),
            Coordinates2D((math.inf, 2)),
            Coordinates2D((1, -math.inf)),
            Coordinates2D((2, math.nan)),
        )
        self.empty = Empty()

    def test_properties_coords_invalid(self) -> None:
        """Tests domain properties when coordinates are valid."""
        for coords in self.invalid_coords:
            self.assertFalse(self.empty.are_inside(coords))


class Domain2DPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.coords = Coordinates2D((1, 2))
        self.r2 = R2
        self.empty = Empty()

    def test_properties_coords_valid(self) -> None:
        """Tests domain properties when coordinates are valid."""
        self.assertTrue(self.r2.are_inside(self.coords))
        self.r2.assert_inside(self.coords)

    def test_properties_coords_invalid(self) -> None:
        """Tests domain properties when coordinates are valid."""
        self.assertFalse(self.empty.are_inside(self.coords))
        with self.assertRaises(OutOfDomainError):
            self.empty.assert_inside(self.coords)


if __name__ == "__main__":
    unittest.main()
