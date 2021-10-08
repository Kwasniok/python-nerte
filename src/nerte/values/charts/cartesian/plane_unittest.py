# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import itertools
import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.coordinates_unittest import coordinates_3d_equiv
from nerte.values.linalg import (
    AbstractVector,
    ZERO_VECTOR,
    UNIT_VECTOR0,
    UNIT_VECTOR1,
    UNIT_VECTOR2,
)
from nerte.values.linalg_unittest import vec_equiv
from nerte.values.interval import Interval
from nerte.values.domains import OutOfDomainError, CartesianProduct2D
from nerte.values.charts.cartesian.plane import Plane


class PlaneConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        interval = Interval(-math.inf, 4.0)
        self.domain = CartesianProduct2D(interval, interval)
        self.v0 = ZERO_VECTOR
        self.b0 = UNIT_VECTOR0
        self.b1 = UNIT_VECTOR1
        self.offset = UNIT_VECTOR2

    def test_constructor(self) -> None:
        """Tests the constructor."""
        Plane(self.b0, self.b1)
        Plane(self.b0, self.b1, domain=self.domain)
        Plane(self.b0, self.b1, offset=self.offset)
        Plane(self.b0, self.b1, domain=self.domain, offset=self.offset)
        with self.assertRaises(ValueError):
            Plane(self.v0, self.b1)
        with self.assertRaises(ValueError):
            Plane(self.b0, self.v0)
        with self.assertRaises(ValueError):
            Plane(self.v0, self.v0)


class FinitePlaneTest(BaseTestCase):
    def setUp(self) -> None:
        self.v1 = UNIT_VECTOR0 * 3
        self.v2 = UNIT_VECTOR1 * 2
        interval = Interval(-1.0, +1.0)
        domain = CartesianProduct2D(interval, interval)
        inside = (0.0,)
        outside = (-2.0, 2.0, -math.inf, math.inf, math.nan)
        values = tuple(itertools.chain(inside, outside))
        self.coords_inside = tuple(
            Coordinates2D((x, y)) for x in inside for y in inside
        )
        self.coords_inside_embedded = tuple(
            Coordinates3D((x * 3, y * 2, 5)) for x in inside for y in inside
        )
        self.coords_outside = tuple(
            Coordinates2D((x, y))
            for x in values
            for y in values
            if not (x in inside and y in inside)
        )
        self.chart = Plane(
            self.v1,
            self.v2,
            domain=domain,
            offset=AbstractVector((0.0, 0.0, 5.0)),
        )

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
            v1, v2 = self.chart.tangential_space(coords)
            self.assertPredicate2(vec_equiv, v1, self.v1)
            self.assertPredicate2(vec_equiv, v2, self.v2)

    def test_tangential_space_raises(self) -> None:
        """Tests tangential space raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.tangential_space(coords)


class SemiFinitePlaneTest(BaseTestCase):
    def setUp(self) -> None:
        self.v1 = UNIT_VECTOR0 * 3
        self.v2 = UNIT_VECTOR1 * 2
        self.n = UNIT_VECTOR2
        interval = Interval(-math.inf, +1.0)
        domain = CartesianProduct2D(interval, interval)
        inside = (0.0, -2.0)
        outside = (2.0, -math.inf, math.inf, math.nan)
        values = tuple(itertools.chain(inside, outside))
        self.coords_inside = tuple(
            Coordinates2D((x, y)) for x in inside for y in inside
        )
        self.coords_inside_embedded = tuple(
            Coordinates3D((x * 3, y * 2, 5)) for x in inside for y in inside
        )
        self.coords_outside = tuple(
            Coordinates2D((x, y))
            for x in values
            for y in values
            if not (x in inside and y in inside)
        )
        self.chart = Plane(
            self.v1,
            self.v2,
            domain=domain,
            offset=AbstractVector((0.0, 0.0, 5.0)),
        )

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
            v1, v2 = self.chart.tangential_space(coords)
            self.assertPredicate2(vec_equiv, v1, self.v1)
            self.assertPredicate2(vec_equiv, v2, self.v2)

    def test_tangential_space_raises(self) -> None:
        """Tests tangential space raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.tangential_space(coords)

    def test_surface_normal(self) -> None:
        """Tests surface normal."""
        for coords in self.coords_inside:
            v = self.chart.surface_normal(coords)
            self.assertPredicate2(vec_equiv, v, self.n)

    def test_surface_normal_raises(self) -> None:
        """Tests surface normal raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.surface_normal(coords)


if __name__ == "__main__":
    unittest.main()
