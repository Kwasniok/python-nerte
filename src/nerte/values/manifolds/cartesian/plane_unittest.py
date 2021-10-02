# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.coordinates_unittest import coordinates_3d_equiv
from nerte.values.interval import Interval
from nerte.values.linalg import AbstractVector, cross
from nerte.values.linalg_unittest import vec_equiv
from nerte.values.util.convert import (
    coordinates_as_vector,
    vector_as_coordinates,
)
from nerte.values.manifolds.base import OutOfDomainError
from nerte.values.manifolds.cartesian.plane import Plane


class PlaneConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.domain = Interval(-1.0, 4.0)
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.v2 = AbstractVector((0.0, 1.0, 0.0))
        self.offset = AbstractVector((0.0, 0.0, 0.0))

    def test_plane_constructor(self) -> None:
        """Tests plane constroctor."""
        Plane(b0=self.v1, b1=self.v2)
        Plane(b0=self.v1, b1=self.v2, offset=self.offset)
        # no zero vector allowed
        with self.assertRaises(ValueError):
            Plane(self.v0, self.v1)
        with self.assertRaises(ValueError):
            Plane(self.v1, self.v0)
        with self.assertRaises(ValueError):
            Plane(self.v0, self.v0)
        # no linear dependency allowed
        with self.assertRaises(ValueError):
            Plane(self.v1, self.v1)


class PlaneDomainTest(BaseTestCase):
    def setUp(self) -> None:
        v1 = AbstractVector((1.0, 0.0, 0.0))
        v2 = AbstractVector((0.0, 1.0, 0.0))
        self.finite_plane = Plane(
            v1, v2, x0_domain=Interval(-1.0, 2.0), x1_domain=Interval(3.0, -4.0)
        )
        self.infinite_plane = Plane(v1, v2)
        self.coords = (
            Coordinates2D((-2.0, -2.0)),
            Coordinates2D((3.0, -2.0)),
            Coordinates2D((1.0, -5.0)),
            Coordinates2D((1.0, 4.0)),
        )

    def test_plane_embed_domain(self) -> None:
        """Tests plane's embedding."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_plane.embed(coords)
        for coords in self.coords:
            self.infinite_plane.embed(coords)

    def test_plane_surface_normal_domain(self) -> None:
        """Tests plane's surface normal."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_plane.surface_normal(coords)
        for coords in self.coords:
            self.infinite_plane.surface_normal(coords)

    def test_plane_tangential_space_domain(self) -> None:
        """Tests plane's tangential space."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_plane.tangential_space(coords)
        for coords in self.coords:
            self.infinite_plane.tangential_space(coords)


class PlanePropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.v2 = AbstractVector((0.0, 1.0, 0.0))
        self.n = cross(self.v1, self.v2)
        self.offsets = (
            AbstractVector((0.0, 0.0, 0.0)),
            AbstractVector((1.1, 2.2, 3.3)),
        )
        self.planes = tuple(
            Plane(self.v1, self.v2, offset=o) for o in self.offsets
        )
        c2d_0 = Coordinates2D((0.0, 0.0))
        c2d_1 = Coordinates2D((1.0, 0.0))
        c2d_2 = Coordinates2D((0.0, 1.0))
        c2d_3 = Coordinates2D((2.0, -3.0))
        c3d_0 = Coordinates3D((0.0, 0.0, 0.0))
        c3d_1 = Coordinates3D((1.0, 0.0, 0.0))
        c3d_2 = Coordinates3D((0.0, 1.0, 0.0))
        c3d_3 = Coordinates3D((2.0, -3.0, 0.0))
        self.coords_2d = (c2d_0, c2d_1, c2d_2, c2d_3)
        self.coords_3d = (c3d_0, c3d_1, c3d_2, c3d_3)

    def test_plane_embed(self) -> None:
        """Tests plane coordinates."""
        for plane, offset in zip(self.planes, self.offsets):
            for c2d, c3d in zip(self.coords_2d, self.coords_3d):
                self.assertPredicate2(
                    coordinates_3d_equiv,
                    plane.embed(c2d),
                    vector_as_coordinates(coordinates_as_vector(c3d) + offset),
                )

    def test_plane_surface_normal(self) -> None:
        """Tests plane's surface normal."""
        for plane in self.planes:
            for c2d in self.coords_2d:
                self.assertPredicate2(
                    vec_equiv,
                    plane.surface_normal(c2d),
                    self.n,
                )

    def test_plane_tangential_space(self) -> None:
        """Tests plane's tangential space."""
        for plane in self.planes:
            for c2d in self.coords_2d:
                b0, b1 = plane.tangential_space(c2d)
                self.assertPredicate2(vec_equiv, b0, self.v1)
                self.assertPredicate2(vec_equiv, b1, self.v2)


if __name__ == "__main__":
    unittest.main()
