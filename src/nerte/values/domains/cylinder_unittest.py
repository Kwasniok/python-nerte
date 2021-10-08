# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math
import itertools

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.domains.cylinder import CYLINDER


class CylinderPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        r_inside = (1e-9, 1.0)
        phi_inside = (-math.pi / 2, 0.0, +math.pi / 2)
        z_inside = (-1e9, 0, +1e9)
        r_outside = (0.0, math.inf, math.nan)
        phi_outside = (
            -math.pi,
            +math.pi,
            -math.inf,
            math.inf,
            math.nan,
        )
        z_outside = (-math.inf, +math.inf, math.nan)
        r_values = tuple(itertools.chain(r_inside, r_outside))
        phi_values = tuple(itertools.chain(phi_inside, phi_outside))
        z_values = tuple(itertools.chain(z_inside, z_outside))
        self.valid_coords = tuple(
            Coordinates3D((r, phi, z))
            for r in r_inside
            for phi in phi_inside
            for z in z_inside
        )
        self.invalid_coords = tuple(
            Coordinates3D((r, phi, z))
            for r in r_values
            for phi in phi_values
            for z in z_values
            if not (r in r_inside and phi in phi_inside and z in z_inside)
        )

    def test_properties_coords_valid(self) -> None:
        """Tests coordinate membership."""
        for coords in self.valid_coords:
            self.assertTrue(CYLINDER.are_inside(coords))

    def test_properties_coords_invalid(self) -> None:
        """Tests coordinate membership."""
        for coords in self.invalid_coords:
            self.assertFalse(CYLINDER.are_inside(coords))


if __name__ == "__main__":
    unittest.main()
