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
from nerte.values.linalg import (
    AbstractVector,
    UNIT_VECTOR0,
    UNIT_VECTOR1,
    AbstractMatrix,
)
from nerte.values.linalg_unittest import vec_equiv
from nerte.values.domains import OutOfDomainError, CartesianProduct2D, R3
from nerte.values.submanifolds.plane import Plane
from nerte.values.transitions.transition_3d import (
    IdentityTransition3D,
)
from nerte.values.transitions.linear_3d import Linear3D
from nerte.values.submanifolds.submanifold_2d_in_3d import (
    CanonicalImmersion2DIn3D,
)
from nerte.values.submanifolds.pushforward_submanifold_2d_in_3d import (
    PushforwardSubmanifold2DIn3D,
)


class ConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.submanifold = Plane(UNIT_VECTOR0, UNIT_VECTOR1)
        self.transition = IdentityTransition3D()

    def test_constructor(self) -> None:
        """Tests the constructor."""
        manifold = PushforwardSubmanifold2DIn3D(
            self.submanifold, self.transition
        )
        self.assertIs(manifold.submanifold, self.submanifold)
        self.assertIs(manifold.transition, self.transition)


class PropertiesTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        interval = Interval(-1.0, +1.0)
        domain = CartesianProduct2D(interval, interval)
        immersion = CanonicalImmersion2DIn3D(domain)
        matrix = AbstractMatrix(
            AbstractVector((2.0, 3.0, 5.0)),
            AbstractVector((7.0, 11.0, 13.0)),
            AbstractVector((17.0, 19.0, 23.0)),
        )
        transition = Linear3D(R3, R3, matrix)
        self.submanifold = PushforwardSubmanifold2DIn3D(immersion, transition)
        inside = (-0.5, 0.0, +0.5)
        outside = (-2.0, 2.0, math.inf, math.nan)
        values = tuple(itertools.chain(inside, outside))
        self.coords_inside = tuple(
            Coordinates2D((x, y)) for x in inside for y in inside
        )
        self.coords_inside_embedded = tuple(
            Coordinates3D((2 * x + 3 * y, 7 * x + 11 * y, 17 * x + 19 * y))
            for x in inside
            for y in inside
        )
        self.coords_outside = tuple(
            Coordinates2D((x, y))
            for x in values
            for y in values
            if not (x in inside and y in inside)
        )
        self.tangential_space = (
            AbstractVector((2, 7, 17)),
            AbstractVector((3, 11, 19)),
        )
        self.surface_normal = AbstractVector((5, 13, 23))

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
        b0, b1 = self.tangential_space
        for coords in self.coords_inside:
            v0, v1 = self.submanifold.tangential_space(coords)
            self.assertPredicate2(vec_equiv, v0, b0)
            self.assertPredicate2(vec_equiv, v1, b1)

    def test_tangential_space_raises(self) -> None:
        """Tests tangential space raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.submanifold.tangential_space(coords)

    def test_surface_normal(self) -> None:
        """Tests surface normal."""
        for coords in self.coords_inside:
            v = self.submanifold.surface_normal(coords)
            self.assertPredicate2(vec_equiv, v, self.surface_normal)

    def test_surface_normal_raises(self) -> None:
        """Tests surface normal raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.submanifold.surface_normal(coords)


if __name__ == "__main__":
    unittest.main()
