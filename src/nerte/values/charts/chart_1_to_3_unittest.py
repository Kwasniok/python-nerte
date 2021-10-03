# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates1D, Coordinates3D
from nerte.values.coordinates_unittest import coordinates_3d_equiv
from nerte.values.interval import Interval
from nerte.values.linalg import UNIT_VECTOR0
from nerte.values.linalg_unittest import vec_equiv
from nerte.values.domains import OutOfDomainError, CartesianProduct1D
from nerte.values.charts.chart_1_to_3 import CanonicalImmersionChart1DTo3D


class CanonicalImmersionChart2DTo3DTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        interval = Interval(-1.0, +1.0)
        domain = CartesianProduct1D(interval)
        inside = (0.0,)
        outside = (-2.0, 2.0, math.inf, math.nan)
        self.coords_inside = tuple(Coordinates1D((x,)) for x in inside)
        self.coords_inside_embedded = tuple(
            Coordinates3D((x, 0, 0)) for x in inside
        )
        self.coords_outside = tuple(Coordinates1D((x,)) for x in outside)
        self.chart = CanonicalImmersionChart1DTo3D(domain)

    def test_embed(self) -> None:
        """Tests coordinate embedding."""
        for coords, coords_embedded in zip(
            self.coords_inside, self.coords_inside_embedded
        ):
            c = self.chart.embed(coords)
            self.assertPredicate2(coordinates_3d_equiv, c, coords_embedded)

    def test_embed_raises(self) -> None:
        """Tests coordinate embedding raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.embed(coords)

    def test_tangential_space(self) -> None:
        """Tests tangential space."""
        for coords in self.coords_inside:
            v = self.chart.tangential_space(coords)
            self.assertPredicate2(vec_equiv, v, UNIT_VECTOR0)

    def test_tangential_space_raises(self) -> None:
        """Tests tangential space raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.tangential_space(coords)


if __name__ == "__main__":
    unittest.main()
