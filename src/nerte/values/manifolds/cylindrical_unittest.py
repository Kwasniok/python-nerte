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
from nerte.values.domain import Domain1D
from nerte.values.linalg import AbstractVector, dot, are_linear_dependent
from nerte.values.linalg_unittest import scalar_equiv, vec_equiv
from nerte.values.manifold import OutOfDomainError
from nerte.values.manifolds.cylindrical import Plane
from nerte.values.util.convert import cylindric_to_carthesian_vector


class PlaneConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.domain = Domain1D(-1.0, 4.0)
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
            v1, v2, x0_domain=Domain1D(-1.0, 2.0), x1_domain=Domain1D(3.0, -4.0)
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
    # pylint: disable=R0902
    def setUp(self) -> None:
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.v2 = AbstractVector((0.0, 1.0, 0.0))
        self.offset = AbstractVector((0.0, 0.0, 0.0))
        self.plane = Plane(self.v1, self.v2, offset=self.offset)
        c2d_1 = Coordinates2D((1.0, 0.0))
        c2d_2 = Coordinates2D((0.0, 1.0))
        c2d_3 = Coordinates2D((2.0, -3.0))
        c3d_1 = Coordinates3D((1.0, 0.0, 0.0))
        c3d_2 = Coordinates3D((1.0, math.pi / 2, 0.0))
        c3d_3 = Coordinates3D((math.sqrt(13), math.atan2(-3.0, 2.0), 0.0))
        self.coords_2d = (c2d_1, c2d_2, c2d_3)
        self.coords_3d = (c3d_1, c3d_2, c3d_3)
        self.n = AbstractVector((0.0, 0.0, 1.0))
        self.n_cartesian = AbstractVector((0.0, 0.0, 1.0))

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
        for c2d in self.coords_2d:
            self.assertPredicate2(
                vec_equiv,
                self.plane.surface_normal(c2d),
                self.n,
            )

    def test_plane_tangential_space(self) -> None:
        """Tests plane's tangential space."""
        for c2d, c3d in zip(self.coords_2d, self.coords_3d):
            b0, b1 = self.plane.tangential_space(c2d)
            # must be two linear independent vectors
            self.assertFalse(are_linear_dependent((b0, b1)))
            # which are orthogonal to the normal vector
            v0 = cylindric_to_carthesian_vector(c3d, b0)
            v1 = cylindric_to_carthesian_vector(c3d, b1)
            self.assertPredicate2(
                scalar_equiv,
                dot(self.n_cartesian, v0),
                0.0,
            )
            self.assertPredicate2(
                scalar_equiv,
                dot(self.n_cartesian, v1),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
