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
from nerte.values.interval import Interval
from nerte.values.linalg import UNIT_VECTOR0, UNIT_VECTOR1, UNIT_VECTOR2
from nerte.values.linalg_unittest import vec_equiv
from nerte.values.domains import OutOfDomainError, CartesianProduct2D
from nerte.values.submanifolds.submanifold_2d_in_3d import (
    CanonicalImmersion2DIn3D,
)


class CanonicalImmersionChart2DTo3DTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        interval = Interval(-1.0, +1.0)
        domain = CartesianProduct2D(interval, interval)
        inside = (0.0,)
        outside = (-2.0, 2.0, math.inf, math.nan)
        values = tuple(itertools.chain(inside, outside))
        self.coords_inside = tuple(
            Coordinates2D((x, y)) for x in inside for y in inside
        )
        self.coords_inside_embedded = tuple(
            Coordinates3D((x, y, 0)) for x in inside for y in inside
        )
        self.coords_outside = tuple(
            Coordinates2D((x, y))
            for x in values
            for y in values
            if not (x in inside and y in inside)
        )
        self.submanifold = CanonicalImmersion2DIn3D(domain)

    def test_embed(self) -> None:
        """Tests coordinate embedding."""
        for coords, coords_embedded in zip(
            self.coords_inside, self.coords_inside_embedded
        ):
            c = self.submanifold.embed(coords)
            self.assertPredicate2(coordinates_3d_equiv, c, coords_embedded)

    def test_embed_raises(self) -> None:
        """Tests coordinate embedding raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.submanifold.embed(coords)

    def test_tangential_space(self) -> None:
        """Tests tangential space."""
        for coords in self.coords_inside:
            v0, v1 = self.submanifold.tangential_space(coords)
            self.assertPredicate2(vec_equiv, v0, UNIT_VECTOR0)
            self.assertPredicate2(vec_equiv, v1, UNIT_VECTOR1)

    def test_tangential_space_raises(self) -> None:
        """Tests tangential space raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.submanifold.tangential_space(coords)

    def test_surface_normal(self) -> None:
        """Tests surface normal."""
        for coords in self.coords_inside:
            v = self.submanifold.surface_normal(coords)
            self.assertPredicate2(vec_equiv, v, UNIT_VECTOR2)

    def test_surface_normal_raises(self) -> None:
        """Tests surface normal raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.submanifold.surface_normal(coords)


if __name__ == "__main__":
    unittest.main()
