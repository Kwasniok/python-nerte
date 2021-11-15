# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.domains.cylindrical_swirl import CylindricalSwirlDomain


class DomainTest(BaseTestCase):
    def setUp(self) -> None:
        swirl = 7
        self.domain = CylindricalSwirlDomain(swirl)
        self.coords_inside = (
            Coordinates3D((1, 0, 0)),
            Coordinates3D((1, 0 - 14, +2)),
            Coordinates3D((1, 0 + 14, -2)),
            Coordinates3D((1e-14, 0, 0)),
            Coordinates3D((1e-14, 0 - 14e-14, 2)),
            Coordinates3D((1e-14, 0 + 14e-14, -2)),
            Coordinates3D((1, -math.pi + 1e-9, 0)),
            Coordinates3D((1, -math.pi - 14 + 1e-9, +2)),
            Coordinates3D((1, -math.pi + 14 + 1e-9, -2)),
            Coordinates3D((1, +math.pi - 1e-9, 0)),
            Coordinates3D((1, +math.pi - 14 - 1e-9, +2)),
            Coordinates3D((1, +math.pi + 14 - 1e-9, -2)),
        )
        self.coords_outside = (
            Coordinates3D((0, 0, 0)),
            Coordinates3D((1, -math.pi, 0)),
            Coordinates3D((1, +math.pi, 0)),
            Coordinates3D((1, -math.pi - 14, +2)),
            Coordinates3D((1, +math.pi - 14, -2)),
            Coordinates3D((-math.inf, 0, 0)),
            Coordinates3D((+math.inf, 0, 0)),
            Coordinates3D((math.nan, 0, 0)),
            Coordinates3D((1, -math.inf, 0)),
            Coordinates3D((1, +math.inf, 0)),
            Coordinates3D((1, math.nan, 0)),
            Coordinates3D((1, 0, -math.inf)),
            Coordinates3D((1, 0, +math.inf)),
            Coordinates3D((1, 0, math.nan)),
        )

    def test_domain_inside(self) -> None:
        """Test the coordinates inside the domain."""
        for coords in self.coords_inside:
            self.assertTrue(self.domain.are_inside(coords))

    def test_domain_outside(self) -> None:
        """Test the coordinates outside the domain."""
        for coords in self.coords_outside:
            self.assertFalse(self.domain.are_inside(coords))


if __name__ == "__main__":
    unittest.main()
