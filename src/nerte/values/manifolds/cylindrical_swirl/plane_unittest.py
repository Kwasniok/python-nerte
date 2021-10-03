# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.coordinates_unittest import coordinates_3d_equiv
from nerte.values.tangential_vector import TangentialVector
from nerte.values.interval import Interval
from nerte.values.linalg import AbstractVector
from nerte.values.linalg_unittest import vec_equiv
from nerte.values.manifolds import OutOfDomainError
from nerte.values.manifolds.cylindrical_swirl.plane import Plane
from nerte.values.transformations.cylindrical_cylindrical_swirl import (
    cylindrical_to_cylindrical_swirl_coords,
    cylindrical_to_cylindrical_swirl_vector,
)
from nerte.values.transformations.cartesian_cylindrical import (
    cartesian_to_cylindrical_coords,
    cartesian_to_cylindrical_vector,
)


class PlaneConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirls = (0.0, 1.0, -1.0)
        self.invalid_swirls = (math.nan, math.inf, -math.inf)
        self.domain = Interval(-1.0, 4.0)
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.v2 = AbstractVector((0.0, 1.0, 0.0))
        self.offset = AbstractVector((0.0, 0.0, 0.0))

    def test_plane_constructor(self) -> None:
        """Tests plane constroctor."""
        for swirl in self.swirls:
            Plane(swirl=swirl, b0=self.v1, b1=self.v2)
            Plane(swirl=swirl, b0=self.v1, b1=self.v2, offset=self.offset)
            # no zero vector allowed
            with self.assertRaises(ValueError):
                Plane(swirl, self.v0, self.v1)
            with self.assertRaises(ValueError):
                Plane(swirl, self.v1, self.v0)
            with self.assertRaises(ValueError):
                Plane(swirl, self.v0, self.v0)
            # no linear dependency allowed
            with self.assertRaises(ValueError):
                Plane(swirl, self.v1, self.v1)
        # invalid swirl
        for swirl in self.invalid_swirls:
            with self.assertRaises(ValueError):
                Plane(swirl=swirl, b0=self.v1, b1=self.v2)


class PlaneDomainTest(BaseTestCase):
    def setUp(self) -> None:
        swirl = 1.0
        v1 = AbstractVector((1.0, 0.0, 0.0))
        v2 = AbstractVector((0.0, 1.0, 0.0))
        self.finite_plane = Plane(
            swirl,
            v1,
            v2,
            x0_domain=Interval(-1.0, 2.0),
            x1_domain=Interval(3.0, -4.0),
        )
        self.infinite_plane = Plane(swirl, v1, v2)
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
    # pylint: disable=R0902
    def setUp(self) -> None:
        self.swirl = 1 / 17
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.v2 = AbstractVector((0.0, 1.0, 0.0))
        self.offset = AbstractVector((2.0, 3.0, 5.0))
        self.plane = Plane(self.swirl, self.v1, self.v2, offset=self.offset)
        self.coords_2d = (
            Coordinates2D((0.0, 0.0)),
            Coordinates2D((1.0, 0.0)),
            Coordinates2D((0.0, 1.0)),
            Coordinates2D((2.0, -3.0)),
        )
        carth_coords_3d = (
            Coordinates3D((2.0, 3.0, 5.0)),
            Coordinates3D((2.0 + 1.0, 3.0, 5.0)),
            Coordinates3D((2.0, 3.0 + 1.0, 5.0)),
            Coordinates3D((2.0 + 2.0, 3.0 - 3.0, 5.0)),
        )
        self.coords_3d = tuple(
            cylindrical_to_cylindrical_swirl_coords(
                self.swirl, cartesian_to_cylindrical_coords(c3d)
            )
            for c3d in carth_coords_3d
        )
        self.n_cartesian = AbstractVector((0.0, 0.0, 1.0))
        self.ns = tuple(
            cylindrical_to_cylindrical_swirl_vector(
                self.swirl,
                cartesian_to_cylindrical_vector(
                    TangentialVector(c3d, self.n_cartesian)
                ),
            ).vector
            for c3d in carth_coords_3d
        )
        carth_tangential_space = (self.v1, self.v2)
        self.tangential_spaces = tuple(
            tuple(
                cylindrical_to_cylindrical_swirl_vector(
                    self.swirl,
                    cartesian_to_cylindrical_vector(TangentialVector(c3d, v)),
                ).vector
                for v in carth_tangential_space
            )
            for c3d in carth_coords_3d
        )

    def test_plane_embed(self) -> None:
        """Tests plane coordinates."""
        for c2d, c3d in zip(self.coords_2d, self.coords_3d):
            self.assertPredicate2(
                coordinates_3d_equiv,
                self.plane.embed(c2d),
                c3d,
            )

    def test_plane_surface_normal(self) -> None:
        """Tests plane's surface normal."""
        for c2d, n in zip(self.coords_2d, self.ns):
            self.assertPredicate2(
                vec_equiv,
                self.plane.surface_normal(c2d),
                n,
            )

    def test_plane_tangential_space(self) -> None:
        """Tests plane's tangential space."""
        for c2d, (v0, v1) in zip(self.coords_2d, self.tangential_spaces):
            b0, b1 = self.plane.tangential_space(c2d)
            self.assertPredicate2(
                vec_equiv,
                b0,
                v0,
            )
            self.assertPredicate2(
                vec_equiv,
                b1,
                v1,
            )


if __name__ == "__main__":
    unittest.main()