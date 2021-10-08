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
from nerte.values.linalg import (
    AbstractVector,
    ZERO_VECTOR,
    UNIT_VECTOR0,
    UNIT_VECTOR1,
)
from nerte.values.linalg_unittest import vec_equiv
from nerte.values.domains import OutOfDomainError, CartesianProduct1D
from nerte.values.charts.cartesian.line import Line


class LineConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        interval = Interval(-math.inf, 4.0)
        self.domain = CartesianProduct1D(interval)
        self.v0 = ZERO_VECTOR
        self.v1 = UNIT_VECTOR0
        self.offset = UNIT_VECTOR1

    def test_constructor(self) -> None:
        """Tests the constructor."""
        Line(direction=self.v1)
        Line(direction=self.v1, domain=self.domain)
        Line(direction=self.v1, offset=self.offset)
        Line(direction=self.v1, domain=self.domain, offset=self.offset)
        with self.assertRaises(ValueError):
            Line(self.v0)


class FiniteLineTest(BaseTestCase):
    def setUp(self) -> None:
        self.v = UNIT_VECTOR0 * 3
        interval = Interval(-1.0, +1.0)
        domain = CartesianProduct1D(interval)
        inside = (0.0,)
        outside = (-2.0, 2.0, -math.inf, math.inf, math.nan)
        self.coords_inside = tuple(Coordinates1D((x,)) for x in inside)
        self.coords_inside_embedded = tuple(
            Coordinates3D((x * 3, 0, 5)) for x in inside
        )
        self.coords_outside = tuple(Coordinates1D((x,)) for x in outside)
        self.chart = Line(
            self.v, domain=domain, offset=AbstractVector((0.0, 0.0, 5.0))
        )

    def test_embed(self) -> None:
        """Tests coordinate embedding."""
        for coords, coords_embedded in zip(
            self.coords_inside, self.coords_inside_embedded
        ):
            c = self.chart.embed(coords)
            self.assertPredicate2(coordinates_3d_equiv, c, coords_embedded)

    def test_tangential_space(self) -> None:
        """Tests tangential space."""
        for coords in self.coords_inside:
            v = self.chart.tangential_space(coords)
            self.assertPredicate2(vec_equiv, v, self.v)

    def test_tangential_space_raises(self) -> None:
        """Tests tangential space raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.tangential_space(coords)


class SemifiniteLineTest(BaseTestCase):
    def setUp(self) -> None:
        self.v = UNIT_VECTOR0 * 3
        interval = Interval(-math.inf, +1.0)
        domain = CartesianProduct1D(interval)
        inside = (-2.0, 0.0)
        outside = (2.0, -math.inf, math.inf, math.nan)
        self.coords_inside = tuple(Coordinates1D((x,)) for x in inside)
        self.coords_inside_embedded = tuple(
            Coordinates3D((x * 3, 0, 0)) for x in inside
        )
        self.coords_outside = tuple(Coordinates1D((x,)) for x in outside)
        self.chart = Line(self.v, domain=domain)

    def test_embed(self) -> None:
        """Tests coordinate embedding."""
        for coords, coords_embedded in zip(
            self.coords_inside, self.coords_inside_embedded
        ):
            c = self.chart.embed(coords)
            self.assertPredicate2(coordinates_3d_equiv, c, coords_embedded)

    def test_tangential_space(self) -> None:
        """Tests tangential space."""
        for coords in self.coords_inside:
            v = self.chart.tangential_space(coords)
            self.assertPredicate2(vec_equiv, v, self.v)

    def test_tangential_space_raises(self) -> None:
        """Tests tangential space raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.tangential_space(coords)


class InfiniteLineTest(BaseTestCase):
    def setUp(self) -> None:
        self.v = UNIT_VECTOR0 * 3
        inside = (-2.0, 0.0, 2.0)
        outside = (-math.inf, math.inf, math.nan)
        self.coords_inside = tuple(Coordinates1D((x,)) for x in inside)
        self.coords_inside_embedded = tuple(
            Coordinates3D((x * 3, 0, 0)) for x in inside
        )
        self.coords_outside = tuple(Coordinates1D((x,)) for x in outside)
        self.chart = Line(self.v)

    def test_embed(self) -> None:
        """Tests coordinate embedding."""
        for coords, coords_embedded in zip(
            self.coords_inside, self.coords_inside_embedded
        ):
            c = self.chart.embed(coords)
            self.assertPredicate2(coordinates_3d_equiv, c, coords_embedded)

    def test_tangential_space(self) -> None:
        """Tests tangential space."""
        for coords in self.coords_inside:
            v = self.chart.tangential_space(coords)
            self.assertPredicate2(vec_equiv, v, self.v)

    def test_tangential_space_raises(self) -> None:
        """Tests tangential space raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.tangential_space(coords)


if __name__ == "__main__":
    unittest.main()
