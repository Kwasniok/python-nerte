# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates1D, Coordinates3D
from nerte.values.coordinates_unittest import coordinates_3d_equiv
from nerte.values.interval import Interval
from nerte.values.linalg import AbstractVector
from nerte.values.linalg_unittest import vec_equiv
from nerte.values.util.convert import (
    coordinates_as_vector,
    vector_as_coordinates,
)
from nerte.values.manifolds.base import OutOfDomainError
from nerte.values.manifolds.cartesian.line import Line


class LineConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.domain = Interval(-1.0, 4.0)
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.offset = AbstractVector((0.0, 0.0, 0.0))

    def test_plane_constructor(self) -> None:
        """Tests plane constroctor."""
        Line(direction=self.v1)
        Line(direction=self.v1, offset=self.offset)
        with self.assertRaises(ValueError):
            Line(self.v0)


class LineDomainTest(BaseTestCase):
    def setUp(self) -> None:
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.finite_line = Line(self.v1, Interval(-1.0, 2.0))
        self.infinite_line = Line(self.v1)
        self.coords = (Coordinates1D((-2.0,)), Coordinates1D((3.0,)))
        self.coords = (Coordinates1D((-2.0,)), Coordinates1D((3.0,)))

    def test_line_embed_domain(self) -> None:
        """Tests line coordinates."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_line.embed(coords)
        for coords in self.coords:
            self.infinite_line.embed(coords)

    def test_line_tangential_space_domain(self) -> None:
        """Tests line's tangential space."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_line.tangential_space(coords)
        for coords in self.coords:
            self.infinite_line.tangential_space(coords)


class LinePropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.v = AbstractVector((1.0, 2.0, 3.0))
        self.offsets = (
            AbstractVector((0.0, 0.0, 0.0)),
            AbstractVector((1.1, 2.2, 3.3)),
        )
        self.lines = tuple(Line(self.v, offset=o) for o in self.offsets)
        c1d_0 = Coordinates1D((0.0,))
        c1d_1 = Coordinates1D((-1.0,))
        c1d_3 = Coordinates1D((2.0,))
        c3d_0 = Coordinates3D((0.0, 0.0, 0.0))
        c3d_1 = Coordinates3D((-1.0, -2.0, -3.0))
        c3d_3 = Coordinates3D((2.0, 4.0, 6.0))
        self.coords_1d = (c1d_0, c1d_1, c1d_3)
        self.coords_3d = (c3d_0, c3d_1, c3d_3)

    def test_line_embed(self) -> None:
        """Tests line's embedding."""
        for line, offset in zip(self.lines, self.offsets):
            for c1d, c3d in zip(self.coords_1d, self.coords_3d):
                self.assertPredicate2(
                    coordinates_3d_equiv,
                    line.embed(c1d),
                    vector_as_coordinates(coordinates_as_vector(c3d) + offset),
                )

    def test_linee_tangential_space(self) -> None:
        """Tests line's tangential space."""
        for line in self.lines:
            for c1d in self.coords_1d:
                b = line.tangential_space(c1d)
                self.assertPredicate2(vec_equiv, b, self.v)


if __name__ == "__main__":
    unittest.main()
