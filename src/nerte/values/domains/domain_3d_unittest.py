# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.domains.base import OutOfDomainError
from nerte.values.domains.domain_3d import R3, Empty


class R3PropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.valid_coords = (Coordinates3D((1, 2, 3)),)
        self.invalid_coords = (
            Coordinates3D((math.inf, 2, 3)),
            Coordinates3D((1, -math.inf, 3)),
            Coordinates3D((1, 2, math.nan)),
        )
        self.r3 = R3

    def test_properties_coords_valid(self) -> None:
        """Tests domain properties when coordinates are valid."""
        for coords in self.valid_coords:
            self.assertTrue(self.r3.are_inside(coords))

    def test_properties_coords_invalid(self) -> None:
        """Tests domain properties when coordinates are valid."""
        for coords in self.invalid_coords:
            self.assertFalse(self.r3.are_inside(coords))


class EmptyPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.invalid_coords = (
            Coordinates3D((1, 2, 3)),
            Coordinates3D((math.inf, 2, 3)),
            Coordinates3D((1, -math.inf, 3)),
            Coordinates3D((1, 2, math.nan)),
        )
        self.empty = Empty()

    def test_properties_coords_invalid(self) -> None:
        """Tests domain properties when coordinates are valid."""
        for coords in self.invalid_coords:
            self.assertFalse(self.empty.are_inside(coords))


class Domain3DPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.coords = Coordinates3D((1, 2, 3))
        self.r3 = R3
        self.empty = Empty()

    def test_properties_coords_valid(self) -> None:
        """Tests domain properties when coordinates are valid."""
        self.assertTrue(self.r3.are_inside(self.coords))
        self.r3.assert_inside(self.coords)

    def test_properties_coords_invalid(self) -> None:
        """Tests domain properties when coordinates are valid."""
        self.assertFalse(self.empty.are_inside(self.coords))
        with self.assertRaises(OutOfDomainError):
            self.empty.assert_inside(self.coords)


if __name__ == "__main__":
    unittest.main()
