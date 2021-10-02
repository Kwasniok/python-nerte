# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates1D
from nerte.values.domains.base import OutOfDomainError
from nerte.values.domains.domain_1d import R1, Empty


class R1PropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.valid_coords = (Coordinates1D((1,)),)
        self.invalid_coords = (
            Coordinates1D((math.inf,)),
            Coordinates1D((-math.inf,)),
            Coordinates1D((math.nan,)),
        )
        self.r1 = R1

    def test_properties_coords_valid(self) -> None:
        """Tests domain properties when coordinates are valid."""
        for coords in self.valid_coords:
            self.assertTrue(self.r1.are_inside(coords))

    def test_properties_coords_invalid(self) -> None:
        """Tests domain properties when coordinates are valid."""
        for coords in self.invalid_coords:
            self.assertFalse(self.r1.are_inside(coords))


class EmptyPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.invalid_coords = (
            Coordinates1D((1,)),
            Coordinates1D((math.inf,)),
            Coordinates1D((-math.inf,)),
            Coordinates1D((math.nan,)),
        )
        self.empty = Empty()

    def test_properties_coords_invalid(self) -> None:
        """Tests domain properties when coordinates are valid."""
        for coords in self.invalid_coords:
            self.assertFalse(self.empty.are_inside(coords))


class Domain1DPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.coords = Coordinates1D((1,))
        self.r1 = R1
        self.empty = Empty()

    def test_properties_coords_valid(self) -> None:
        """Tests domain properties when coordinates are valid."""
        self.assertTrue(self.r1.are_inside(self.coords))
        self.r1.assert_inside(self.coords)

    def test_properties_coords_invalid(self) -> None:
        """Tests domain properties when coordinates are valid."""
        self.assertFalse(self.empty.are_inside(self.coords))
        with self.assertRaises(OutOfDomainError):
            self.empty.assert_inside(self.coords)


if __name__ == "__main__":
    unittest.main()
